import numpy as np
import torch
import torch.nn as nn
import math



class FourierFeatureEncoding(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_bands=16, max_freq=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Create a bank of frequencies: 2^0, 2^1, ..., 2^L
        freq_bands = 2.0 ** torch.linspace(0.0, math.log2(max_freq), num_bands)
        self.register_buffer('freq_bands', freq_bands)

        # Project from (input_dim * num_bands * 2) -> hidden_dim
        # *2 because we use both Sin and Cos
        self.output_dim = input_dim * num_bands * 2
        self.proj = nn.Linear(self.output_dim, hidden_dim)

    def forward(self, x):
        # x: [B, N, 3]
        # 1. Compute frequencies
        # [B, N, 3, 1] * [num_bands] -> [B, N, 3, num_bands]
        x_expanded = x.unsqueeze(-1) * self.freq_bands

        # 2. Reshape to apply sin/cos
        # [B, N, 3 * num_bands]
        x_flat = x_expanded.view(x.shape[0], x.shape[1], -1)

        # 3. Apply Sin/Cos features
        # [B, N, 3 * num_bands * 2]
        features = torch.cat([torch.sin(x_flat), torch.cos(x_flat)], dim=-1)

        # 4. Project to model dimension
        return self.proj(features)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN with residual connections."""

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, key_padding_mask=None):
        """
        Args:
            query: [B, Lq, H]
            key: [B, Lk, H] — used as both key and value
            key_padding_mask: [B, Lk] bool (True = ignore)
        Returns:
            [B, Lq, H]
        """
        q = self.norm_q(query)
        kv = self.norm_kv(key)
        h, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        query = query + self.dropout1(h)

        h = self.ffn(self.norm_ffn(query))
        query = query + self.dropout2(h)
        return query


class PerceiverTokenizer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_latents=256,
                 num_heads=4, num_encoder_layers=6, num_decoder_layers=2,
                 dropout=0.0, stress_head=False, n_input_frames=1):
        super().__init__()
        self.n_input_frames = n_input_frames

        # 1. Geometry Encoder (Fourier Features)
        self.pos_encoding = FourierFeatureEncoding(input_dim=3, hidden_dim=hidden_dim)

        # 2. Physics Encoder (Input Displacement)
        # When n_input_frames > 1, encoder accepts concatenated frames
        # (e.g., 3 frames × 3 channels = 9), but decoder always outputs input_dim (3).
        self.input_proj = nn.Linear(input_dim * n_input_frames, hidden_dim)

        # 3. Latent Array (The "Bottleneck")
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # 4. Encoder: cross-attention + self-attention stack
        self.cross_attn_encoder = CrossAttentionBlock(
            hidden_dim, num_heads, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder_self_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        # 5. Pre-bottleneck LayerNorm. Name preserved for ckpt key compat.
        self.pre_vq_norm = nn.LayerNorm(hidden_dim)

        # 6. Decoder: cross-attention + optional self-attention stack
        self.cross_attn_decoder = CrossAttentionBlock(
            hidden_dim, num_heads, dropout=dropout)

        if num_decoder_layers > 0:
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True,
                norm_first=True,
            )
            self.decoder_self_attn = nn.TransformerEncoder(
                decoder_layer, num_layers=num_decoder_layers,
                enable_nested_tensor=False,
            )
        else:
            self.decoder_self_attn = None

        # 8. Final projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, input_dim)

        # 9. Gradient prediction head (Sobolev): predicts grad_u as direct output
        # Ref-NeRF pattern — separate head for gradient, shared decoder features
        self.grad_proj = nn.Linear(hidden_dim, input_dim * input_dim)  # H -> 9
        self.grad_input_dim = input_dim
        # Whether grad_proj carried a training signal. The head is built for
        # every tokenizer but only trained under --sobolev-method grad-head;
        # `from_checkpoint` sets this from the saved config, and the engine and
        # `decode_full_mesh` consult it before using the head for stress.
        self.grad_head_trained = True

        # 10. Stress prediction head: predicts Cauchy stress (symmetric, 6 components)
        # Distinct from gradient head — stress head is universal (any constitutive law),
        # gradient head only works when stress = f(grad_u) (e.g., Neo-Hookean).
        self.stress_head_enabled = stress_head
        if stress_head:
            self.stress_proj = nn.Linear(hidden_dim, 6)  # Cauchy sym6: xx,yy,zz,xy,xz,yz

    def forward(self, x, ref_coords, padding_mask=None):
        """
        Args:
            x: [B, N, input_dim*n_input_frames] - Displacement fields (Physics).
                Single-frame: [B, N, 3]. Multi-frame: [B, N, 9] for 3 frames.
            ref_coords: [B, N, 3] - Reference mesh coordinates (Geometry)
            padding_mask: [B, N] - Boolean mask (True where padding exists)
        """
        B = x.shape[0]

        # --- A. Embed Inputs ---
        mesh_features = self.input_proj(x) + self.pos_encoding(ref_coords)

        # --- B. Perceiver Encoder ---
        latents = self.latents.repeat(B, 1, 1)

        # Cross-attention: latents query mesh features
        z_enc = self.cross_attn_encoder(
            query=latents, key=mesh_features,
            key_padding_mask=padding_mask,
        )

        # Self-attention on latents
        z_enc = self.encoder_self_attn(z_enc)

        # Pre-bottleneck normalization (continuous bottleneck — pass-through).
        z_enc = self.pre_vq_norm(z_enc)

        # --- C. Perceiver Decoder ---
        queries = self.pos_encoding(ref_coords)

        # Cross-attention: mesh queries latents
        features = self.cross_attn_decoder(query=queries, key=z_enc)

        # Decoder self-attention (with padding mask for variable-size meshes)
        if self.decoder_self_attn is not None:
            src_key_padding_mask = padding_mask if padding_mask is not None else None
            features = self.decoder_self_attn(
                features, src_key_padding_mask=src_key_padding_mask)

        # Final projection
        normed = self.final_norm(features)
        x_recon = self.final_proj(normed)

        # Gradient head: predict displacement gradient from shared features
        grad_pred = None
        if hasattr(self, 'grad_proj'):
            grad_pred = self.grad_proj(normed).reshape(
                *features.shape[:-1], self.grad_input_dim, self.grad_input_dim)

        # Stress head: predict Cauchy stress sym6 from shared features
        stress_pred = None
        if self.stress_head_enabled and hasattr(self, 'stress_proj'):
            stress_pred = self.stress_proj(normed)  # [B, N, 6]

        return x_recon, grad_pred, stress_pred

    @torch.no_grad()
    def encode(self, x, ref_coords, padding_mask=None):
        """Encode displacements to continuous latent embeddings.

        Args:
            x: [B, N, input_dim*n_input_frames] - Displacement fields
            ref_coords: [B, N, 3] - Reference mesh coordinates
            padding_mask: [B, N] - Boolean mask (True where padding exists)

        Returns:
            z_enc: [B, K, H] - Continuous latent embeddings
        """
        B = x.shape[0]

        mesh_features = self.input_proj(x) + self.pos_encoding(ref_coords)
        latents = self.latents.repeat(B, 1, 1)

        z_enc = self.cross_attn_encoder(
            query=latents, key=mesh_features,
            key_padding_mask=padding_mask,
        )
        z_enc = self.encoder_self_attn(z_enc)
        z_enc = self.pre_vq_norm(z_enc)
        return z_enc

    @torch.no_grad()
    def decode_embeddings(self, z, ref_coords, padding_mask=None):
        """Decode continuous embeddings to displacements.

        Args:
            z: [B, K, H] - Continuous embeddings
            ref_coords: [B, N, 3] - Reference mesh coordinates
            padding_mask: [B, N] - Boolean mask (True where padding exists)

        Returns:
            x_recon: [B, N, 3] - Reconstructed displacements
        """
        queries = self.pos_encoding(ref_coords)
        features = self.cross_attn_decoder(query=queries, key=z)

        if self.decoder_self_attn is not None:
            src_key_padding_mask = padding_mask if padding_mask is not None else None
            features = self.decoder_self_attn(
                features, src_key_padding_mask=src_key_padding_mask)

        x_recon = self.final_proj(self.final_norm(features))
        return x_recon

    @torch.no_grad()
    def decode_with_stress(self, z, ref_coords, padding_mask=None):
        """Decode continuous embeddings and return displacement + stress predictions.

        Like decode_embeddings but also runs the stress head if enabled.

        Args:
            z: [B, K, H] - Continuous embeddings
            ref_coords: [B, N, 3] - Reference mesh coordinates
            padding_mask: [B, N] - Boolean mask (True where padding exists)

        Returns:
            x_recon: [B, N, 3] - Reconstructed displacements
            stress_pred: [B, N, 6] or None - Predicted Cauchy stress sym6
        """
        queries = self.pos_encoding(ref_coords)
        features = self.cross_attn_decoder(query=queries, key=z)

        if self.decoder_self_attn is not None:
            src_key_padding_mask = padding_mask if padding_mask is not None else None
            features = self.decoder_self_attn(
                features, src_key_padding_mask=src_key_padding_mask)

        normed = self.final_norm(features)
        x_recon = self.final_proj(normed)
        stress_pred = None
        if self.stress_head_enabled and hasattr(self, 'stress_proj'):
            stress_pred = self.stress_proj(normed)
        return x_recon, stress_pred

    @staticmethod
    def grad_head_trained_from_config(cfg) -> bool:
        """True when the checkpoint's training run supervised grad_proj
        (--sobolev-method grad-head with sobolev_beta > 0). Checkpoints that
        predate these config keys are assumed trained."""
        if 'sobolev_method' not in cfg:
            return True
        beta = cfg.get('sobolev_beta', 0.0)
        return cfg.get('sobolev_method') == 'grad-head' and float(beta or 0.0) > 0.0

    @classmethod
    def from_checkpoint(cls, path, device='cuda', map_location=None):
        """Load a tokenizer from a saved checkpoint.

        Returns (tokenizer, stats) where stats is a dict with auxiliary numerics
        the caller needs (`disp_scale`, optional `stress_mean`/`stress_std`, full
        `config`). The tokenizer is on `device`, in eval mode.
        """
        from leia.training import load_checkpoint
        state_dict, _opt, ckpt = load_checkpoint(path, map_location=map_location or device)
        cfg = ckpt.get('config', {})

        model = cls(
            input_dim=3,
            hidden_dim=cfg['hidden_dim'],
            num_latents=cfg['num_latents'],
            num_heads=cfg.get('num_heads', 4),
            num_encoder_layers=cfg['num_encoder_layers'],
            num_decoder_layers=cfg['num_decoder_layers'],
            dropout=cfg.get('dropout', 0.0),
            stress_head=cfg.get('stress_head', False),
            n_input_frames=cfg.get('n_input_frames', 1),
        ).to(device)
        # Extra keys from legacy checkpoints (e.g. `vq.*`) are tolerated;
        # missing keys are not — they would leave layers randomly initialised.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"{path}: checkpoint lacks {len(missing)} parameter(s) that this "
                f"config needs, e.g. {missing[:3]}; it does not match its own config")
        model.eval()
        model.grad_head_trained = cls.grad_head_trained_from_config(cfg)

        stats = {
            'config': cfg,
            'disp_scale': float(ckpt.get('disp_scale', 1.0)),
        }
        # Cauchy sym6 normalization stats (stress-head path). Attached to the
        # model so `decode_full_mesh` / `compute_predicted_vm` can denormalize
        # without three callers re-implementing the same 8-line attach.
        sm = ckpt.get('stress_mean')
        ss = ckpt.get('stress_std')
        if sm is not None and ss is not None:
            sm_arr = sm if isinstance(sm, (list, tuple)) else np.asarray(sm).tolist()
            ss_arr = ss if isinstance(ss, (list, tuple)) else np.asarray(ss).tolist()
            model._stress_mean = torch.tensor(sm_arr, dtype=torch.float32, device=device)
            model._stress_std = torch.tensor(ss_arr, dtype=torch.float32, device=device)
            stats['stress_mean'] = sm
            stats['stress_std'] = ss
        else:
            model._stress_mean = None
            model._stress_std = None
        return model, stats


def subsample_fps(fps_order, subsample_n, fps_core_n, rng=None):
    """Pick subsample_n indices: deterministic FPS core + a remainder drawn
    without replacement from the FPS tail. Without an explicit `rng` the
    remainder uses seed 42, i.e. it is the same node set on every call (the
    training dataset calls it this way, so the subsample is fixed per plate).

    Parameters
    ----------
    fps_order : (N,) int — node indices in farthest-point-sampled order.
    subsample_n : int — total samples to return. Returns the full array if
        N <= subsample_n.
    fps_core_n : int — first `fps_core_n` indices of fps_order are kept
        deterministically; the rest are filled by random sampling without
        replacement from the FPS tail.
    rng : np.random.Generator or None — defaults to seed 42 for reproducibility.

    Returns
    -------
    indices : (min(N, subsample_n),) int — selected node indices.
    """
    N = len(fps_order)
    if N <= subsample_n:
        return fps_order.copy()
    core_n = min(fps_core_n, subsample_n)
    core = fps_order[:core_n]
    n_rand = subsample_n - core_n
    if n_rand > 0:
        remainder = fps_order[core_n:]
        if rng is None:
            rng = np.random.default_rng(42)
        rand_idx = rng.choice(len(remainder), size=n_rand, replace=False)
        return np.concatenate([core, remainder[rand_idx]])
    return core


@torch.no_grad()
def decode_full_mesh(tokenizer, z_enc, ref_full_t, disp_scale, device,
                     chunk_size=50000, coord_scale=None,
                     grad_head_convention='B'):
    """Decode latents at every node of a full mesh, in chunks.

    Stress-head priority: if the tokenizer has a stress head with normalization
    stats, returns Cauchy sym6 + von Mises per node. Else, if it has a gradient
    head and `coord_scale` is provided, returns physical-space ∂u/∂x. Otherwise,
    returns only displacement.

    `grad_head_convention='B'` (j-axis, the production convention) divides by
    `coord_scale` on the last gradient axis. `'A'` divides on the middle axis
    and is kept for older grad-head checkpoints.

    Returns
    -------
    u_phys : (N, 3) float64 — physical displacement
    sym6_node : (N, 6) float64 or None — Cauchy stress sym6 per node
    vm_node : (N,) float32 or None — von Mises derived from sym6
    grad_u_phys : (N, 3, 3) float64 or None — physical grad_u (grad-head path)
    """
    N_full = ref_full_t.shape[1]
    z_batch = z_enc.unsqueeze(0)
    has_stress_head = (getattr(tokenizer, 'stress_head_enabled', False)
                       and getattr(tokenizer, '_stress_mean', None) is not None)
    has_grad_head = ((not has_stress_head) and hasattr(tokenizer, 'grad_proj')
                     and getattr(tokenizer, 'grad_head_trained', True))

    u_parts = []
    sym6_parts = [] if has_stress_head else None
    grad_parts = [] if has_grad_head else None

    for start in range(0, N_full, chunk_size):
        end = min(start + chunk_size, N_full)
        ref_chunk = ref_full_t[:, start:end, :]
        queries = tokenizer.pos_encoding(ref_chunk)
        features = tokenizer.cross_attn_decoder(query=queries, key=z_batch)
        if tokenizer.decoder_self_attn is not None:
            features = tokenizer.decoder_self_attn(features)
        normed = tokenizer.final_norm(features)
        u_parts.append(tokenizer.final_proj(normed).squeeze(0).cpu().numpy())
        if has_stress_head:
            stress_norm = tokenizer.stress_proj(normed).squeeze(0)
            sigma_sym6 = (stress_norm * tokenizer._stress_std[None, :]
                          + tokenizer._stress_mean[None, :])
            sym6_parts.append(sigma_sym6.cpu().numpy())
        elif has_grad_head:
            gp = tokenizer.grad_proj(normed).reshape(1, end - start, 3, 3)
            grad_parts.append(gp.squeeze(0).cpu().numpy())

    u_phys = (np.concatenate(u_parts, axis=0) * disp_scale).astype(np.float64)

    sym6_node = None
    vm_node = None
    if has_stress_head:
        from leia.metrics import vm_from_sym6
        sym6_node = np.concatenate(sym6_parts, axis=0).astype(np.float64)
        vm_node = vm_from_sym6(sym6_node).astype(np.float32)

    grad_u_phys = None
    if has_grad_head and coord_scale is not None:
        grad_all = np.concatenate(grad_parts, axis=0)
        if grad_head_convention == 'A':
            cs_inv = coord_scale[None, :, None]
        else:
            cs_inv = coord_scale[None, None, :]
        grad_u_phys = (grad_all * disp_scale / cs_inv).astype(np.float64)

    return u_phys, sym6_node, vm_node, grad_u_phys
