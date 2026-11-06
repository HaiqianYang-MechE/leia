"""Dynamics transformer (AdaLN-conditioned, continuous-latent).

Production model: AdaLNContinuousDynamicsTransformer.

Dataset classes live in `leia.datasets`. This file is model code only.
"""

import torch
import torch.nn as nn


# Production architectural defaults. These are not hyperparameters that
# typically vary per run — they're the AdaLN backbone shape that all
# current paper checkpoints share. Override at construction if needed.
DYNAMICS_HEADS = 8
DYNAMICS_LAYERS = 12
DYNAMICS_DROPOUT = 0.3


class AdaLNTransformerLayer(nn.Module):
    """Transformer encoder layer with FiLM conditioning (gamma/beta only).

    Modulates LayerNorm output with condition-dependent scale and shift:
        h = gamma * LayerNorm(x) + beta
        x = x + Attn(h)           [attention block]
        h = gamma * LayerNorm(x) + beta
        x = x + MLP(h)            [MLP block]

    Unlike AdaLN-Zero, there are no alpha gates — the transformer layers
    always contribute to the output. This prevents the "identity trap"
    where the model learns to keep gates near zero.

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer" (AAAI 2018).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # FiLM: project conditioning → 4 * d_model modulation params
        # (gamma1, beta1, gamma2, beta2)
        self.film_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model),
        )
        # Init: gamma bias=0 (→ scale=1+0=1), beta bias=0 (→ shift=0)
        # So initial behavior is standard pre-norm transformer
        nn.init.zeros_(self.film_proj[-1].bias)
        # Small weight init for gradual modulation
        nn.init.normal_(self.film_proj[-1].weight, std=0.02)

    def forward(self, x, cond):
        """
        Args:
            x: [B, S, D] — sequence features
            cond: [B, D] — conditioning vector (from action MLP)
        Returns:
            x: [B, S, D] — updated features
        """
        params = self.film_proj(cond)  # [B, 4*D]
        g1, b1, g2, b2 = params.chunk(4, dim=-1)  # each [B, D]

        # --- Attention block ---
        h = self.norm1(x)
        h = h * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        h = self.dropout1(h)
        x = x + h

        # --- MLP block ---
        h = self.norm2(x)
        h = h * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        h = self.mlp(h)
        x = x + h

        return x


class AdaLNContinuousDynamicsTransformer(nn.Module):
    """Dynamics transformer with FiLM action conditioning (continuous latent).

    Instead of prepending action as prefix tokens (which get diluted by
    K state tokens in self-attention), this model modulates every transformer
    layer's LayerNorm with action-dependent scale/shift.

    Uses FiLM (gamma/beta) without alpha gates to avoid the "identity
    trap" where the model keeps gates near zero and ignores actions.

    Forward: predicts the next latent embedding from the current latent +
    action (and optionally the previous latent, when velocity_input=True).
    """

    def __init__(self, num_latents, embed_dim,
                 action_dim=4, heads=DYNAMICS_HEADS,
                 layers=DYNAMICS_LAYERS, dropout=DYNAMICS_DROPOUT,
                 residual=False, velocity_input=False):
        super().__init__()

        self.num_latents = num_latents
        self.residual = residual
        self.velocity_input = velocity_input

        if velocity_input:
            # Concatenate [state, velocity] → project to embed_dim
            self.vel_proj = nn.Linear(2 * embed_dim, embed_dim)
        else:
            self.state_proj = nn.Linear(embed_dim, embed_dim)

        # Action conditioning MLP
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Positional embedding for state tokens only (no prefix tokens)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_latents, embed_dim) * 0.02
        )

        # AdaLN transformer layers
        self.layers = nn.ModuleList([
            AdaLNTransformerLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
            )
            for _ in range(layers)
        ])

        # Final norm + output head (continuous latent → continuous latent)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, state, action, prev_state=None):
        """
        Args:
            state: [B, K, H] float — current latent
            action: [B, action_dim] float
            prev_state: [B, K, H] float, optional — previous latent for velocity
        Returns:
            Predicted next-frame latent [B, K, H].
        """
        if self.velocity_input:
            vel = state - prev_state if prev_state is not None else torch.zeros_like(state)
            s_emb = self.vel_proj(torch.cat([state, vel], dim=-1))  # [B, K, H]
        else:
            s_emb = self.state_proj(state)                          # [B, K, H]
        cond = self.action_proj(action)                              # [B, H]

        x = s_emb + self.pos_embed                                   # [B, K, H]
        for layer in self.layers:
            x = layer(x, cond)
        x = self.final_norm(x)
        out = self.head(x)                                           # [B, K, H]
        if self.residual:
            out = state + out                                         # delta prediction
        return out

    @classmethod
    def from_checkpoint(cls, path, tokenizer_stats=None, device='cuda',
                        map_location=None, heads=DYNAMICS_HEADS, layers=DYNAMICS_LAYERS):
        """Load dynamics from a saved checkpoint.

        Args:
            path: dynamics checkpoint path.
            tokenizer_stats: stats dict returned by `PerceiverTokenizer.from_checkpoint`.
                Used to derive `embed_dim` and `num_latents`.
            heads, layers: backbone shape. Defaults match production; override only
                if your ckpt was trained with non-default values.

        Returns:
            (model, stats) where stats includes config, action_dim, action_scale.
        """
        from leia.training import load_checkpoint
        state_dict, _opt, ckpt = load_checkpoint(path, map_location=map_location or device)
        cfg = ckpt.get('config', {})

        if tokenizer_stats is not None:
            tok_cfg = tokenizer_stats.get('config', tokenizer_stats)
            embed_dim = tok_cfg['hidden_dim']
            num_latents = tok_cfg['num_latents']
        else:
            embed_dim = cfg['embed_dim']
            num_latents = cfg['num_latents']

        action_dim = ckpt.get('action_dim', cfg.get('action_dim', 4))

        model = cls(
            num_latents=num_latents,
            embed_dim=embed_dim,
            action_dim=action_dim,
            heads=heads,
            layers=layers,
            dropout=cfg.get('dropout', DYNAMICS_DROPOUT),
            residual=cfg.get('residual', False),
            velocity_input=cfg.get('velocity_input', False),
        ).to(device)
        # Extra keys from legacy checkpoints (e.g. discrete-mode `state_embed.*`)
        # are tolerated; missing keys are not — they would leave layers random.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"{path}: checkpoint lacks {len(missing)} parameter(s) that this "
                f"config needs, e.g. {missing[:3]}; it does not match its own config")
        model.eval()

        stats = {
            'config': cfg,
            'action_dim': action_dim,
            'action_scale': ckpt.get('action_scale', cfg.get('action_scale', 1.0)),
            'action_key': cfg.get('action_key', 'cumulative_bc'),
        }
        return model, stats
