"""Inference engine for the LEIA design-search runtime.

Thin driver around `leia.engine.LeiaEngine`. Owns runtime/session state —
per-mesh normalization, cumulative BC, prevq, prev_state, boundary geometry
for F-D curves — and delegates pure model wiring (encode / dynamics step /
decode / stress) to `LeiaEngine`.
"""

import os
import sys
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

from leia.engine import LeiaEngine
from leia.metrics import vm_from_sym6
from leia.stress_torch import neo_hookean_P_torch
from leia.stress import (
    tet4_dNdx,
    identify_right_boundary_nodes,
    precompute_boundary_geometry,
    neo_hookean_P,
    assemble_internal_force,
    generalized_force_from_f_int,
    boundary_reaction_force_from_cauchy,
)

from design_search.core.config import ServingConfig
from design_search.core.meshing import MeshResult


@dataclass
class Session:
    session_id: str
    rve_id: str
    mesh: MeshResult
    ref_norm_full: np.ndarray   # (N, 3) normalized coords for ALL mesh nodes
    ref_norm_sub: np.ndarray    # (N_sub, 3) normalized coords for encoding
    coord_scale: np.ndarray     # (3,) half-range
    c_min: np.ndarray           # (3,) min coords
    prevq: torch.Tensor         # (K, H) on GPU — current latent
    cum_bc: np.ndarray          # (4,) cumulative BCs
    step: int = 0
    created_at: float = field(default_factory=time.time)
    # Previous-step latent for velocity-conditioned dynamics. At step 0 this
    # equals prevq (system at rest, velocity = 0). Only consumed when the
    # loaded dynamics ckpt has velocity_input=True; otherwise unused.
    prev_state: torch.Tensor = None
    # Boundary geometry for F-D curve computation
    bdy_geom: dict = field(default_factory=dict)
    right_node_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    cy: float = 0.0  # y-center of right face
    cz: float = 0.0  # z-center of right face
    graph_data: dict = field(default_factory=dict)  # original RVE graph for mutations


@dataclass
class StepResult:
    session_id: str
    step: int
    displacement: np.ndarray    # (N, 3) physical displacement
    von_mises: np.ndarray       # (N,) von Mises stress
    cum_bc: np.ndarray          # (4,) current cumulative BC
    gen_force: np.ndarray       # (4,) [Fx, Mx, Fy, Fz] reaction forces
    gen_disp: np.ndarray        # (4,) [dx, theta, dy, dz] cumulative disp
    inference_time_s: float


class ModelEngine:
    """Driver for design-search runtime.

    Wraps `LeiaEngine` for model inference; owns per-session state (mesh,
    normalization, cumulative BC, boundary geometry). The model layer is
    stateless across sessions; this class adds the runtime around it.
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.device = torch.device(config.device)
        print(f"Loading models on {self.device}...")
        self.engine = LeiaEngine(
            tokenizer_ckpt=config.tokenizer_ckpt,
            dynamics_ckpt=config.dynamics_ckpt,
            device=self.device,
        )
        # Convenience proxies — used by design_search/landscape/score.py and
        # the grad-head fallback in step_batch.
        self.tokenizer = self.engine.tokenizer
        self.disp_scale = self.engine.disp_scale
        self.tok_cfg = self.engine.tok_cfg
        self._velocity_input = self.engine.velocity_input
        print(f"  Tokenizer: K={self.engine.K}, H={self.engine.H}, "
              f"disp_scale={self.disp_scale:.4f}, "
              f"stress_head={self.engine.has_stress_head}, "
              f"grad_head={self.engine.has_grad_head}")
        print(f"  Dynamics: velocity_input={self._velocity_input}")
        print("Models loaded.")

    # ------------------------------------------------------------------
    # OOD Detection
    # ------------------------------------------------------------------

    def compute_recon_error(self, session: 'Session') -> float:
        """Round-trip reconstruction MSE — delegates to LeiaEngine."""
        return self.engine.compute_recon_error(session.ref_norm_sub)

    # ------------------------------------------------------------------
    # Session creation
    # ------------------------------------------------------------------

    def create_session(self, mesh: MeshResult, rve_id: str = "") -> Session:
        """Create a new session from a plate mesh.

        Computes normalization, encodes frame 0 (zero displacement) via the
        library engine, and precomputes boundary geometry for F-D curves.
        """
        points = mesh.points.astype(np.float32)
        N = len(points)

        # FPS-like subsampling: for serving, use random permutation
        # (FPS is too slow, and frame 0 encoding quality is trivially good)
        subsample_n = min(10240, N)
        rng = np.random.default_rng(42)
        sub_idx = rng.permutation(N)[:subsample_n]
        coords_sub = points[sub_idx]

        # Normalization must match training, which maps the FULL mesh's
        # bounding box to [-1, 1]; the random subsample misses the extreme
        # nodes, so its box would push the outermost full-mesh query nodes
        # slightly outside the trained range.
        c_min = points.min(axis=0)
        c_max = points.max(axis=0)
        coord_range = np.maximum(c_max - c_min, 1e-8)
        coord_scale = coord_range / 2.0

        ref_norm_sub = ((coords_sub - c_min) / coord_range * 2 - 1).astype(np.float32)
        ref_norm_full = ((points - c_min) / coord_range * 2 - 1).astype(np.float32)

        # Encode frame 0 (zero physical disp) via the library engine.
        zero_disp = np.zeros_like(ref_norm_sub)
        prevq = self.engine.encode_frame(zero_disp, ref_norm_sub)

        # Precompute boundary geometry for F-D curves (CPU, pure numpy)
        points_f64 = mesh.points.astype(np.float64)
        dNdx, volumes = tet4_dNdx(points_f64, mesh.cells)
        right_node_ids, _ = identify_right_boundary_nodes(points_f64, eps=0.1)
        bdy_geom = precompute_boundary_geometry(mesh.cells, dNdx, volumes, right_node_ids)
        right_coords = points_f64[right_node_ids]
        cy = float(right_coords[:, 1].mean())
        cz = float(right_coords[:, 2].mean())

        session = Session(
            session_id=str(uuid.uuid4())[:8],
            rve_id=rve_id,
            mesh=mesh,
            ref_norm_full=ref_norm_full,
            ref_norm_sub=ref_norm_sub,
            coord_scale=coord_scale,
            c_min=c_min,
            prevq=prevq,
            # Frame-0 velocity is zero, so prev_state == prevq.
            prev_state=prevq.clone(),
            cum_bc=np.zeros(4, dtype=np.float32),
            bdy_geom=bdy_geom,
            right_node_ids=right_node_ids,
            cy=cy,
            cz=cz,
        )
        return session

    # ------------------------------------------------------------------
    # Batched dynamics step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_batch(self, sessions: list[Session], action: list[int]) -> list[StepResult]:
        """Run one dynamics step for multiple sessions with the same action.

        Driver responsibilities (kept here):
          - Per-session BC accumulation
          - Batched (prevq, prev_state, ref) tensor assembly
          - Per-plate reaction-force computation on boundary tets
        Model responsibilities (delegated to LeiaEngine):
          - Single AR step
          - bf16 decode + stress dispatch (returns physical disp + sym6)
        """
        if not sessions:
            return []

        B = len(sessions)
        speeds = np.array(self.config.speeds, dtype=np.float32)
        action_arr = np.array(action, dtype=np.float32)
        device = self.device
        mu = self.config.mu
        lmbda = self.config.lmbda

        # Update cumulative BCs (driver-side: action_scale + speeds are
        # design-search conventions, not part of the model layer).
        for s in sessions:
            s.cum_bc = s.cum_bc + action_arr * speeds

        # --- Batched dynamics step (delegate) ---
        prevq_batch = torch.stack([s.prevq for s in sessions])         # (B, K, H)
        bc_batch = torch.stack([
            torch.tensor(s.cum_bc * self.config.action_scale,
                         dtype=torch.float32, device=device)
            for s in sessions
        ])                                                             # (B, 4)
        if self._velocity_input:
            prev_state_batch = torch.stack([s.prev_state for s in sessions])
            pred_batch, _ = self.engine.step_dynamics(
                prevq_batch, bc_batch, prev_state=prev_state_batch)
        else:
            pred_batch, _ = self.engine.step_dynamics(prevq_batch, bc_batch)

        # --- Batched decode (delegate; pad to max-N, one forward pass) ---
        node_counts = [len(s.ref_norm_full) for s in sessions]
        N_max = max(node_counts)

        ref_batch = torch.zeros(B, N_max, 3, dtype=torch.float32, device=device)
        for i, s in enumerate(sessions):
            ref_batch[i, :node_counts[i]] = torch.tensor(
                s.ref_norm_full, dtype=torch.float32, device=device)

        decoded = self.engine.decode(pred_batch, ref_batch, compute_stress=True)
        # `decoded["disp"]` is already in physical units (LeiaEngine.decode
        # multiplies by disp_scale). cauchy_sym6 is denormalized; vm follows.
        u_phys = decoded["disp"]                                       # (B, N_max, 3)
        sym6_all = decoded["cauchy_sym6"]
        grad_all = decoded["grad_u"]
        vm_all = decoded["vm"]

        # --- Stress-dispatch fork (CPU buffers for per-plate force) ---
        u_cpu = u_phys.cpu().numpy()
        if sym6_all is not None:
            # Stress-head path (production v2).
            vm_cpu = vm_all.cpu().numpy()
            sym6_cpu = sym6_all.cpu().to(torch.float64).numpy()        # for F-D
            grad_u_cpu = None
        elif grad_all is not None:
            # Legacy grad-head path: convert raw `grad_u` (normalized
            # tokenizer space) → physical via per-session coord_scale, apply
            # Neo-Hookean PK1 to recover Cauchy → vm. design_search-side
            # because coord_scale is per-mesh; LeiaEngine doesn't see it.
            cs_batch = torch.tensor(
                np.array([s.coord_scale for s in sessions]),
                dtype=torch.float32, device=device)                    # (B, 3)
            # j-axis (convention 'B'), as in decode_full_mesh and the losses.
            grad_u_phys_gpu = (grad_all.reshape(B, N_max, 3, 3)
                               * self.disp_scale
                               / cs_batch[:, None, None, :])
            I = torch.eye(3, device=device, dtype=torch.float32)
            Fe = I + grad_u_phys_gpu
            Fe_flat = Fe.reshape(-1, 3, 3)
            P_flat = neo_hookean_P_torch(Fe_flat, mu, lmbda)
            P = P_flat.reshape(B, N_max, 3, 3)
            J = torch.linalg.det(Fe_flat).reshape(B, N_max, 1, 1)
            J_safe = torch.clamp(J, min=1e-6)
            sigma = (1.0 / J_safe) * torch.matmul(P, Fe.transpose(-2, -1))
            sym6_legacy = torch.stack([
                sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 2, 2],
                sigma[..., 0, 1], sigma[..., 0, 2], sigma[..., 1, 2],
            ], dim=-1)
            vm_cpu = vm_from_sym6(sym6_legacy).cpu().numpy()
            grad_u_cpu = grad_u_phys_gpu.cpu().to(torch.float64).numpy()
            sym6_cpu = None
        else:
            vm_cpu = np.zeros((B, N_max), dtype=np.float32)
            grad_u_cpu = np.zeros((B, N_max, 3, 3), dtype=np.float64)
            sym6_cpu = None

        # --- Per-plate: slice + force (numpy/fp64, boundary tets only) ---
        results = []
        t0 = time.time()
        for i, session in enumerate(sessions):
            N_i = node_counts[i]
            disp = u_cpu[i, :N_i].astype(np.float32)
            vm = vm_cpu[i, :N_i].astype(np.float32)
            if sym6_cpu is not None:
                gen_force = self._compute_reaction_force_from_sym6(
                    session, disp.astype(np.float64), sym6_cpu[i, :N_i])
            elif grad_u_cpu is not None:
                grad_u_phys = grad_u_cpu[i, :N_i].astype(np.float64)
                gen_force = self._compute_reaction_force(session, grad_u_phys)
            else:
                gen_force = np.zeros(4, dtype=np.float64)

            # Roll the prev_state buffer for velocity-input dynamics: today's
            # prevq becomes tomorrow's prev_state. (No-op for v1 dynamics.)
            session.prev_state = session.prevq
            session.prevq = pred_batch[i]
            session.step += 1

            results.append(StepResult(
                session_id=session.session_id,
                step=session.step,
                displacement=disp,
                von_mises=vm,
                cum_bc=session.cum_bc.copy(),
                gen_force=gen_force,
                gen_disp=session.cum_bc.copy(),
                inference_time_s=(time.time() - t0) / max(i + 1, 1),
            ))

        return results

    # ------------------------------------------------------------------
    # Reaction force computation (CPU, pure numpy)
    # ------------------------------------------------------------------

    def _compute_reaction_force(self, session: Session,
                                grad_u_phys: np.ndarray) -> np.ndarray:
        """Compute [Fx, Mx, Fy, Fz] from gradient head output.

        v1 path. Reads grad_u in physical space, applies Neo-Hookean
        constitutive law to get PK1, integrates boundary tets.
        """
        bdy = session.bdy_geom
        if not bdy or 'cells_bdy' not in bdy:
            return np.zeros(4, dtype=np.float64)

        cells_bdy = bdy['cells_bdy']
        dNdx_bdy = bdy['dNdx_bdy']
        volumes_bdy = bdy['volumes_bdy']
        n_nodes = len(session.mesh.points)

        # Average per-node grad_u to boundary tets
        grad_u_tet = grad_u_phys[cells_bdy].mean(axis=1)  # (B_bdy, 3, 3)

        # Deformation gradient and PK1 stress
        F_bdy = np.eye(3) + grad_u_tet
        P_bdy = neo_hookean_P(F_bdy, self.config.mu, self.config.lmbda)

        # Assemble internal force at boundary
        f_int = assemble_internal_force(P_bdy, dNdx_bdy, volumes_bdy,
                                        cells_bdy, n_nodes)

        # Extract generalized force
        return generalized_force_from_f_int(
            f_int, session.right_node_ids, session.mesh.points,
            session.cy, session.cz)

    def _compute_reaction_force_from_sym6(self, session: Session,
                                          u_phys: np.ndarray,
                                          sym6: np.ndarray) -> np.ndarray:
        """Compute [Fx, Mx, Fy, Fz] from stress-head Cauchy sym6.

        v2 path. The decoded u provides F at the boundary tets (kinematic
        only); σ_Cauchy comes directly from the trained head, so PK1 = J σ F⁻ᵀ
        without any constitutive law assumption. Universal across constitutive
        laws (works for visco/Arruda-Boyce/etc., not just Neo-Hookean).
        """
        bdy = session.bdy_geom
        if not bdy or 'cells_bdy' not in bdy:
            return np.zeros(4, dtype=np.float64)
        return boundary_reaction_force_from_cauchy(
            u_disp=u_phys,
            cauchy_nodal=sym6,
            coords=session.mesh.points.astype(np.float64),
            n_nodes=len(session.mesh.points),
            right_node_ids=session.right_node_ids,
            cy=session.cy,
            cz=session.cz,
            bdy_geom=bdy,
        )

    # ------------------------------------------------------------------
    # Reset session
    # ------------------------------------------------------------------

    def reset_session(self, session: Session):
        """Reset session to frame 0."""
        zero_disp = np.zeros_like(session.ref_norm_sub)
        session.prevq = self.engine.encode_frame(zero_disp, session.ref_norm_sub)
        session.prev_state = session.prevq.clone()
        session.cum_bc = np.zeros(4, dtype=np.float32)
        session.step = 0
