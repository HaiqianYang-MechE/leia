"""LEIA inference engine — the canonical library entry point for surrogate
prediction without runtime/session state.

`LeiaEngine` loads a tokenizer + dynamics checkpoint pair once and exposes
stateless primitives:

  - encode_frame(disp, ref)               — single-frame init
  - encode_history(disp_history, ref)     — multi-frame TF encode
  - step_dynamics(prevq, action, prev_state=None)
                                          — single AR step
  - decode(prevq, ref, compute_stress)    — bf16 decode + stress dispatch
  - rollout(z0, actions, ref)             — convenience: T-step AR + decode
  - compute_recon_error(ref)              — OOD detection (zero-disp roundtrip)

Session state — mesh, normalization, cumulative BC, prevq, prev_state — lives
in the caller. The thin drivers around this engine are:

  - `design_search.core.engine.ModelEngine`    online interactive use
  - `leia/_eval_pipelines/{eval_e2e,eval_tokenizer}.py`  offline eval pipelines
  - `scripts/{eval_microplate_visco,render_e2e_dashboard}.py`  one-shot eval

Inputs accept either numpy arrays (auto-cast to fp32 on engine.device) or
torch tensors (preserved dtype, moved to engine.device if needed). Outputs
are torch tensors on engine.device — convert to CPU/numpy at the call site.

Disp convention: `encode_*` accept **physical-space** displacement and divide
by disp_scale internally to match the tokenizer's training input space;
`decode` returns physical-space displacement (multiplied by disp_scale before
returning). This matches the conventions in
`leia/_eval_pipelines/eval_e2e.py:encode_frame`.

Channel order for n_input_frames>1 multi-frame input: chronological,
[disp_{t-(F-1)}, ..., disp_{t-1}, disp_t] — the layout used by the tokenizer
training pipeline (see `leia.datasets.MmapMultiStructureDataset.__getitem__`).
"""

import os
import warnings

import numpy as np
import torch

from leia.tokenizer import PerceiverTokenizer
from leia.dynamics import AdaLNContinuousDynamicsTransformer
from leia.metrics import vm_from_sym6


class LeiaEngine:
    """Stateless LEIA inference: tokenizer + dynamics + stress readout.

    Loaded once from checkpoint paths. All methods are pure functions of
    inputs; no internal mutable state across calls. Callers manage session
    state (cum_bc, prevq, mesh) themselves.
    """

    def __init__(
        self,
        tokenizer_ckpt: str,
        dynamics_ckpt: str | None = None,
        device='cuda',
    ):
        """Load tokenizer (required) and dynamics (optional).

        If `dynamics_ckpt=None`, the engine runs in tokenizer-only mode:
        `encode_*`, `decode`, and `compute_recon_error` work normally;
        `step_dynamics` and `rollout` raise. Useful for stress-eval scripts
        that don't run AR rollouts.
        """
        if not os.path.exists(tokenizer_ckpt):
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found: {tokenizer_ckpt}")
        if dynamics_ckpt is not None and not os.path.exists(dynamics_ckpt):
            raise FileNotFoundError(
                f"Dynamics checkpoint not found: {dynamics_ckpt}")

        self._device = torch.device(device)

        # --- Tokenizer ---
        # `from_checkpoint` attaches `_stress_mean` and `_stress_std` to the
        # tokenizer instance when the stress head is enabled. We read them
        # off the tokenizer (don't try to "clean up" by stashing them on
        # this engine — keeps `decode` simple).
        self.tokenizer, tok_stats = PerceiverTokenizer.from_checkpoint(
            tokenizer_ckpt, device=self._device)
        cfg = tok_stats['config']
        self._tok_cfg = cfg
        self._disp_scale = float(tok_stats['disp_scale'])
        # K and H are required ckpt fields. A ckpt without them is malformed
        # — silently falling back to constants would let a wrong-shape model
        # load via load_state_dict(strict=False) and produce silently wrong
        # outputs. Raise early instead.
        if 'num_latents' not in cfg or 'hidden_dim' not in cfg:
            raise ValueError(
                f"Tokenizer checkpoint missing required config keys "
                f"num_latents/hidden_dim: cfg={cfg}")
        self._K = int(cfg['num_latents'])
        self._H = int(cfg['hidden_dim'])
        self._n_input_frames = int(cfg.get('n_input_frames', 1))

        stress_head_cfg = bool(cfg.get('stress_head', False))
        self._has_stress_head = (
            stress_head_cfg
            and self.tokenizer._stress_mean is not None
            and hasattr(self.tokenizer, 'stress_proj')
        )
        # grad_proj exists on every tokenizer; it is a usable stress source
        # only if the training run supervised it (see
        # PerceiverTokenizer.grad_head_trained_from_config).
        self._has_grad_head = (
            hasattr(self.tokenizer, 'grad_proj')
            and self.tokenizer.grad_proj is not None
            and getattr(self.tokenizer, 'grad_head_trained', True)
        )

        # --- Dynamics (optional) ---
        # `AdaLNContinuousDynamicsTransformer.from_checkpoint` pins K, H from
        # `tokenizer_stats` (so we always pass the freshly-loaded tok_stats).
        # A dynamics ckpt with mismatched dims will fail to load.
        # velocity_input=True dynamics requires `prev_state` at every step;
        # missing it collapses velocity to state and the AR rollout diverges.
        # Callers check `engine.velocity_input` to decide whether to track
        # `prev_state`.
        if dynamics_ckpt is not None:
            self.dynamics, dyn_stats = AdaLNContinuousDynamicsTransformer.from_checkpoint(
                dynamics_ckpt, tokenizer_stats=tok_stats, device=self._device)
            self._velocity_input = bool(dyn_stats['config'].get('velocity_input', False))
            self._action_scale = float(dyn_stats.get('action_scale', 1.0))
            self._has_dynamics = True
        else:
            self.dynamics = None
            self._velocity_input = False
            self._action_scale = 1.0
            self._has_dynamics = False

    @property
    def K(self) -> int:
        return self._K

    @property
    def H(self) -> int:
        return self._H

    @property
    def n_input_frames(self) -> int:
        return self._n_input_frames

    @property
    def disp_scale(self) -> float:
        return self._disp_scale

    @property
    def has_stress_head(self) -> bool:
        return self._has_stress_head

    @property
    def has_grad_head(self) -> bool:
        return self._has_grad_head

    @property
    def velocity_input(self) -> bool:
        return self._velocity_input

    @property
    def has_dynamics(self) -> bool:
        return self._has_dynamics

    @property
    def action_scale(self) -> float:
        """The `action_scale` the dynamics checkpoint was trained with (1.0 if
        no dynamics). Drivers multiply `cum_bc` by it before `step_dynamics`;
        a different value silently drives the model off-distribution, so
        check any user flag against this."""
        return self._action_scale

    @property
    def tok_cfg(self) -> dict:
        """Tokenizer config dict from the loaded ckpt (read-only view)."""
        return self._tok_cfg

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def encode_frame(self, disp, ref_coords):
        """Single-frame encode → (K, H) latent.

        Replicates a single physical-space displacement frame across
        n_input_frames channels. The AR-init case passes zero displacement
        and the replication is a no-op.

        Parameters
        ----------
        disp : (N_sub, 3) — physical-space displacement (typically zero at
            frame 0). Normalized internally by disp_scale.
        ref_coords : (N_sub, 3) — coords normalized to [-1, 1].

        Returns
        -------
        (K, H) latent on `self.device`.
        """
        ref_t = self._to_tensor(ref_coords).unsqueeze(0)              # (1, N, 3)
        disp_t = self._to_tensor(disp).unsqueeze(0) / self._disp_scale  # (1, N, 3)
        # Repeat the 3-channel block n_input_frames times along channel axis:
        # gives (1, N, 3*F). For zero-disp this matches the legacy AR-init
        # pattern (`torch.zeros(1, N, 3*F)`); for non-zero disp this is a
        # "rest snapshot" interpretation (every past frame is the current
        # frame). Use `encode_history` for proper TF-style history encoding.
        x_t = disp_t.repeat(1, 1, self._n_input_frames)
        z_enc = self.tokenizer.encode(x_t, ref_t)
        return z_enc.squeeze(0)                                       # (K, H)

    @torch.no_grad()
    def encode_history(self, disp_history, ref_coords):
        """Multi-frame TF-style encode → (K, H) latent.

        Uses the **last n_input_frames** frames of `disp_history`,
        concatenated channel-wise in chronological order. Matches the
        convention in `leia/_eval_pipelines/eval_e2e.py:encode_frame` and the
        tokenizer training pipeline.

        Parameters
        ----------
        disp_history : (T, N_sub, 3) — physical-space displacement, T must
            satisfy T >= n_input_frames. Normalized internally by disp_scale.
        ref_coords : (N_sub, 3) — coords normalized to [-1, 1].

        Returns
        -------
        (K, H) latent on `self.device`.
        """
        ref_t = self._to_tensor(ref_coords).unsqueeze(0)              # (1, N, 3)
        hist_t = self._to_tensor(disp_history) / self._disp_scale      # (T, N, 3)
        T = hist_t.shape[0]
        if T < self._n_input_frames:
            raise ValueError(
                f"disp_history length T={T} < n_input_frames={self._n_input_frames}")
        last_F = hist_t[-self._n_input_frames:]                        # (F, N, 3)
        # Reorder to (1, N, F*3) with channel order
        # [x_{t-(F-1)}, y_{t-(F-1)}, z_{t-(F-1)}, ..., x_t, y_t, z_t]:
        x_t = last_F.permute(1, 0, 2).reshape(1, last_F.shape[1], -1)
        z_enc = self.tokenizer.encode(x_t, ref_t)
        return z_enc.squeeze(0)                                       # (K, H)

    @torch.no_grad()
    def step_dynamics(self, prevq, action, prev_state=None):
        """Single AR step.

        Parameters
        ----------
        prevq : (B, K, H) — current latent.
        action : (B, 4) — already-scaled cumulative BC. Caller is
            responsible for `cum_bc * action_scale`; `engine.action_scale`
            reports the value the checkpoint was trained with.
        prev_state : (B, K, H) | None — required when `velocity_input=True`,
            ignored otherwise.

        Returns
        -------
        (next_prevq, next_prev_state)
            Where `next_prev_state` is the *input* `prevq` (rolled forward),
            so callers can pass it back at step t+1. None when
            `velocity_input=False`.
        """
        if not self._has_dynamics:
            raise RuntimeError(
                "step_dynamics requires a dynamics checkpoint; engine was "
                "constructed in tokenizer-only mode (dynamics_ckpt=None)")
        prevq = self._to_tensor(prevq)
        action = self._to_tensor(action)
        if self._velocity_input:
            if prev_state is None:
                raise ValueError(
                    "step_dynamics: velocity_input=True dynamics requires prev_state")
            prev_state = self._to_tensor(prev_state)
            next_prevq = self.dynamics(prevq, action, prev_state=prev_state)
            return next_prevq, prevq
        else:
            if prev_state is not None:
                # The dynamics model was trained without velocity_input, so
                # `prev_state` is silently ignored. Caller is likely confused
                # about whether to track it — flag loudly.
                warnings.warn(
                    "step_dynamics: prev_state provided but engine.velocity_input"
                    " is False; argument is ignored. Check `engine.velocity_input`"
                    " before threading prev_state.",
                    stacklevel=2,
                )
            next_prevq = self.dynamics(prevq, action)
            return next_prevq, None

    @torch.no_grad()
    def decode(self, prevq, ref_coords, compute_stress: bool = True):
        """Decode latent → physical displacement (+ optional stress).

        bf16-autocast cross-attention decode followed by per-head stress
        readout. Matches the training-time bf16 scope.

        Parameters
        ----------
        prevq : (B, K, H) — latent.
        ref_coords : (B, N_max, 3) — normalized coords.
        compute_stress : bool — populate stress fields when a head is
            available. Set False on hot paths that only need disp.

        Returns
        -------
        dict with keys:
            disp:        (B, N_max, 3) physical-space displacement
            cauchy_sym6: (B, N_max, 6) | None — physical-units Cauchy stress
                          [xx, yy, zz, xy, xz, yz] when has_stress_head and
                          compute_stress.
            grad_u:      (B, N_max, 9) | None — raw grad_u from the gradient
                          head, in NORMALIZED tokenizer space. Caller must
                          convert to physical via `disp_scale / coord_scale`.
                          Populated when has_grad_head, has_stress_head=False,
                          and compute_stress=True.
            vm:          (B, N_max) | None — von Mises stress when sym6 is
                          available; otherwise None.
        """
        prevq = self._to_tensor(prevq)
        ref_coords = self._to_tensor(ref_coords)
        # Device-agnostic autocast: 'cuda' or 'cpu' accepted; cuda:N collapses
        # to 'cuda' for autocast purposes.
        autocast_device = str(self._device).split(":")[0]

        sym6 = None
        grad_u = None
        with torch.autocast(autocast_device, dtype=torch.bfloat16):
            queries = self.tokenizer.pos_encoding(ref_coords)
            features = self.tokenizer.cross_attn_decoder(query=queries, key=prevq)
            if self.tokenizer.decoder_self_attn is not None:
                features = self.tokenizer.decoder_self_attn(features)
            normed = self.tokenizer.final_norm(features)
            disp = self.tokenizer.final_proj(normed).float()
            if compute_stress and self._has_stress_head:
                # Normalized Cauchy sym6 → de-normalize per component (Voigt
                # order: [xx, yy, zz, xy, xz, yz]). The .float() cast happens
                # before the multiply, so de-norm runs in fp32 even though
                # this block sits inside the bf16 autocast scope.
                sym6 = self.tokenizer.stress_proj(normed).float()
                sym6 = (sym6 * self.tokenizer._stress_std[None, None, :]
                        + self.tokenizer._stress_mean[None, None, :])
            elif compute_stress and self._has_grad_head:
                grad_u = self.tokenizer.grad_proj(normed).float()

        # Disp from final_proj is in normalized space; multiply by disp_scale
        # to recover physical units.
        disp = disp * self._disp_scale

        out = {"disp": disp, "cauchy_sym6": sym6, "grad_u": grad_u, "vm": None}
        if sym6 is not None:
            out["vm"] = vm_from_sym6(sym6)
        return out

    @torch.no_grad()
    def rollout(self, z0, actions, ref_coords, compute_stress: bool = True):
        """T-step AR rollout — convenience for eval scripts.

        Loops `step_dynamics` + `decode` for T steps, returning the time
        series of decoded outputs. design_search loops manually because it
        interleaves per-step BC accumulation with mesh-bound force
        computation; this convenience is for single-mesh, full-trajectory
        eval scripts.

        Parameters
        ----------
        z0 : (K, H) — initial latent (e.g. from `encode_frame` or
            `encode_history`).
        actions : (T, 4) — per-step ALREADY-SCALED actions. The engine does
            not know about action_scale.
        ref_coords : (N_max, 3) — normalized coords.
        compute_stress : bool — gate stress computation per-step.

        Returns
        -------
        dict with keys:
            disp:        (T, N_max, 3) physical displacement per step
            cauchy_sym6: (T, N_max, 6) | None
            vm:          (T, N_max) | None
            latents:     (T, K, H) — z_1, z_2, ..., z_T (post-step latents)
        """
        z0 = self._to_tensor(z0)
        actions = self._to_tensor(actions)
        ref_coords = self._to_tensor(ref_coords)
        T = actions.shape[0]
        N = ref_coords.shape[0]

        prevq = z0.unsqueeze(0)                                       # (1, K, H)
        prev_state = z0.unsqueeze(0).clone() if self._velocity_input else None
        ref_b = ref_coords.unsqueeze(0)                                # (1, N, 3)

        disp_out = torch.empty(T, N, 3, dtype=torch.float32, device=self._device)
        # Latents from `step_dynamics` are fp32 regardless of z0's dtype;
        # match that to avoid a silent downcast on bf16/half z0 inputs.
        latents_out = torch.empty(T, self._K, self._H, dtype=torch.float32,
                                  device=self._device)
        sym6_out = None
        vm_out = None
        if compute_stress and self._has_stress_head:
            sym6_out = torch.empty(T, N, 6, dtype=torch.float32, device=self._device)
            vm_out = torch.empty(T, N, dtype=torch.float32, device=self._device)

        for t in range(T):
            action_t = actions[t:t + 1]                                # (1, 4)
            prevq, next_prev_state = self.step_dynamics(
                prevq, action_t, prev_state=prev_state)
            if self._velocity_input:
                prev_state = next_prev_state
            decoded = self.decode(prevq, ref_b, compute_stress=compute_stress)
            disp_out[t] = decoded["disp"].squeeze(0)
            latents_out[t] = prevq.squeeze(0)
            if sym6_out is not None and decoded["cauchy_sym6"] is not None:
                sym6_out[t] = decoded["cauchy_sym6"].squeeze(0)
                vm_out[t] = decoded["vm"].squeeze(0)

        return {
            "disp": disp_out,
            "cauchy_sym6": sym6_out,
            "vm": vm_out,
            "latents": latents_out,
        }

    @torch.no_grad()
    def compute_recon_error(self, ref_coords) -> float:
        """Round-trip reconstruction MSE at frame 0 — OOD detection signal.

        Encodes zero-displacement, decodes, computes MSE. High MSE indicates
        the geometry is poorly represented by the tokenizer (out of
        distribution).

        Parameters
        ----------
        ref_coords : (N_sub, 3) — normalized coords.
        """
        ref_t = self._to_tensor(ref_coords).unsqueeze(0)
        N = ref_t.shape[1]
        x = torch.zeros(1, N, 3 * self._n_input_frames,
                        dtype=torch.float32, device=self._device)
        z_enc = self.tokenizer.encode(x, ref_t)
        x_recon = self.tokenizer.decode_embeddings(z_enc, ref_t)
        return float((x_recon ** 2).mean())

    def _to_tensor(self, x):
        """numpy → fp32 tensor on device; tensor → moved to device (dtype kept)."""
        if isinstance(x, torch.Tensor):
            if x.device != self._device:
                x = x.to(self._device)
            return x
        return torch.tensor(x, dtype=torch.float32, device=self._device)
