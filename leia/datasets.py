"""PyTorch Dataset classes for LEIA training.

Tokenizer training (whole-plate displacement → reconstruction + Sobolev/stress):
- `MmapMultiStructureDataset` + `padding_collate` — production tokenizer dataset
  for MicroPlate lattice. Reads per-plate mmap'd FEM trajectories.
- `ViscoLoadReverseDataset` + `make_visco_load_reverse_datasets` — small fixed-mesh
  visco-LR dataset (Arruda-Boyce 3-branch).

Dynamics training (latent token AR rollout):
- `MmapWholePlateTrajectoryDataset` — per-trajectory mmap windows for pushforward
  training (`--method continuous_pf`, the production path).

The mmap layouts share physical pages across DDP ranks via the OS page cache, which
is what makes 8-GPU DDP viable on hundreds of GB of prevq / curated data.
"""

import json
import os
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from leia.regime import MICROPLATE_LATTICE


class MmapWholePlateTrajectoryDataset(Dataset):
    """Memory-mapped whole-plate dataset, multi-step trajectory windows.

    For pushforward training: returns k+2 consecutive frames (the frame
    before the window, then the k+1 window frames) and k actions, so the
    training loop can chain model predictions over k steps with a velocity
    input. This is the production path (`--method continuous_pf`).
    """

    def __init__(self, mmap_dir, split="train", val_ratio=0.1, seed=42,
                 action_scale=1.0, max_groups=0, action_key='cumulative_bc',
                 rollout_steps=2, exclude_plates=None):
        super().__init__()
        self.action_scale = action_scale
        self.rollout_steps = rollout_steps

        exclude_set = set(exclude_plates or [])
        plate_ids = sorted([
            f.replace('_meta.json', '')
            for f in os.listdir(mmap_dir) if f.endswith('_meta.json')
        ])
        if exclude_set:
            plate_ids = [p for p in plate_ids if p not in exclude_set]
        if max_groups > 0:
            plate_ids = plate_ids[:max_groups]

        self._prevq_maps = []
        self._cumbc_arrays = []
        windows = []  # (plate_idx, traj, t_start)

        detected_action_dim = None

        for plate_idx, pid in enumerate(plate_ids):
            prevq_path = os.path.join(mmap_dir, f'{pid}_prevq.npy')
            cumbc_path = os.path.join(mmap_dir, f'{pid}_cumbc.npy')
            meta_path = os.path.join(mmap_dir, f'{pid}_meta.json')

            with open(meta_path, 'r') as fmeta:
                meta = json.load(fmeta)
            n_traj = meta['n_traj']
            self._n_frames = meta['n_frames']
            n_inc = meta['n_inc']

            prevq = np.load(prevq_path, mmap_mode='r')
            cumbc = np.load(cumbc_path)

            self._prevq_maps.append(prevq)
            self._cumbc_arrays.append(cumbc)

            if detected_action_dim is None:
                detected_action_dim = cumbc.shape[-1]

            for traj in range(n_traj):
                for t in range(n_inc - rollout_steps + 1):
                    windows.append((plate_idx, traj, t))

        self._action_dim = detected_action_dim

        n_total = len(windows)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)
        split_idx = int(n_total * (1 - val_ratio))

        if split == "train":
            sel = indices[:split_idx]
        else:
            sel = indices[split_idx:]

        self.windows = [windows[i] for i in sel]

    @property
    def action_dim(self):
        return self._action_dim

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        plate_idx, traj, t_start = self.windows[idx]
        k = self.rollout_steps
        n_frames = self._n_frames

        # Include previous frame for velocity computation: [z(t-1), z(t), ..., z(t+k)]
        traj_start = traj * n_frames
        prev_idx = traj_start + max(0, t_start - 1)
        main_start = traj_start + t_start
        main_end = main_start + k + 1
        main_slice = np.array(self._prevq_maps[plate_idx][main_start:main_end])
        prev_frame = np.array(self._prevq_maps[plate_idx][prev_idx:prev_idx + 1])
        prevq = np.concatenate([prev_frame, main_slice], axis=0)

        actions = self._cumbc_arrays[plate_idx][traj, t_start:t_start + k]

        return {
            "prevq": torch.from_numpy(prevq),                          # [k+2, K, H]
            "actions": torch.tensor(actions * self.action_scale, dtype=torch.float32),
            "timestep": t_start,
        }


# ---------------------------------------------------------------------------
# MicroPlate visco — load-reverse dataset (Arruda-Boyce + 3-branch viscoelastic)
# ---------------------------------------------------------------------------

_VISCO_HDF5_GROUP = 'solid_plate_visco'
_VISCO_N_FRAMES_TOTAL = 101   # 0 = rest, 1-100 = loaded (50 fwd + 50 rev)
_VISCO_N_USABLE = 100         # skip frame 0


from leia.stress import cauchy_to_sym6  # noqa: E402  (re-export; used in __getitem__ below)


class ViscoLoadReverseDataset(Dataset):
    """MicroPlate visco — load-reverse tokenizer training dataset.

    Single fixed mesh (363 nodes). Each sample is one (trajectory, frame).
    Returns displacement and Cauchy stress sym6 as ground truth target.

    The protocol is 100 action steps (50 forward + 50 exact reverse, stride-5
    dt=0.01s). Matched cumulative-BC points between forward and reverse halves
    have different (u, sigma) due to viscoelastic memory — this is a harder
    history-dependent variant of the simple load protocol.

    HDF5 group `solid_plate_visco` layout:
        coords:        (363, 3)
        topology:      (1200, 4)
        inputs:        (1000, 100, 4) int8 — discrete actions {-1, 0, +1}
        displacements: (1000, 101, 363, 3)
        cauchy_stress: (1000, 101, 363, 3, 3)
    """

    def __init__(self, h5_path, splits_json, split='train',
                 disp_scale=None, stress_mean=None, stress_std=None,
                 n_input_frames=1):
        super().__init__()
        self.split = split
        self.n_input_frames = n_input_frames

        with open(splits_json) as f:
            splits = json.load(f)
        indices = splits[split]

        with h5py.File(h5_path, 'r') as f:
            g = f[_VISCO_HDF5_GROUP]
            coords = g['coords'][:]
            self.topology = g['topology'][:]

            n_traj = len(indices)
            n_nodes = coords.shape[0]

            disps = np.zeros((n_traj, _VISCO_N_FRAMES_TOTAL, n_nodes, 3),
                             dtype=np.float32)
            stress_33 = np.zeros((n_traj, _VISCO_N_FRAMES_TOTAL, n_nodes, 3, 3),
                                 dtype=np.float32)
            for i, idx in enumerate(indices):
                disps[i] = g['displacements'][idx]
                stress_33[i] = g['cauchy_stress'][idx]
                if (i + 1) % 100 == 0:
                    print(f"  Loading {split}: {i+1}/{n_traj}...")

        self.n_traj = n_traj
        self.n_nodes = n_nodes

        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        scale = np.maximum(c_max - c_min, 1e-8)
        self.ref_coords = torch.tensor(
            ((coords - c_min) / scale * 2 - 1).astype(np.float32))

        stress_sym6 = cauchy_to_sym6(stress_33)
        del stress_33

        if disp_scale is not None:
            self.disp_scale = disp_scale
        else:
            all_disp = disps[:, 1:].ravel()
            self.disp_scale = float(np.std(all_disp)) if np.std(all_disp) > 1e-8 else 1.0

        if stress_mean is not None and stress_std is not None:
            self.stress_mean = stress_mean
            self.stress_std = stress_std
        else:
            all_stress = stress_sym6[:, 1:].reshape(-1, 6)
            self.stress_mean = torch.tensor(
                all_stress.mean(axis=0).astype(np.float32))
            self.stress_std = torch.tensor(
                np.maximum(all_stress.std(axis=0), 1e-6).astype(np.float32))

        # Frames 1..100 usable (skip frame 0 rest state)
        self.disps = torch.tensor(disps[:, 1:].astype(np.float32))
        self.stress = torch.tensor(stress_sym6[:, 1:].astype(np.float32))
        self.disp_frame0 = torch.tensor(disps[:, 0].astype(np.float32))

        self.n_usable_frames = _VISCO_N_USABLE

    def __len__(self):
        return self.n_traj * self.n_usable_frames

    def __getitem__(self, idx):
        traj_idx = idx // self.n_usable_frames
        frame_idx = idx % self.n_usable_frames

        disp = self.disps[traj_idx, frame_idx]
        stress = self.stress[traj_idx, frame_idx]
        x_target = disp / self.disp_scale

        if self.n_input_frames == 3:
            if frame_idx >= 2:
                x_fm2 = self.disps[traj_idx, frame_idx - 2] / self.disp_scale
                x_fm1 = self.disps[traj_idx, frame_idx - 1] / self.disp_scale
            elif frame_idx == 1:
                x_fm2 = self.disp_frame0[traj_idx] / self.disp_scale
                x_fm1 = self.disps[traj_idx, 0] / self.disp_scale
            else:
                x_fm2 = self.disp_frame0[traj_idx] / self.disp_scale
                x_fm1 = self.disp_frame0[traj_idx] / self.disp_scale
            x_input = torch.cat([x_fm2, x_fm1, x_target], dim=-1)
        else:
            x_input = x_target

        return {
            'x': x_input,
            'x_target': x_target,
            'ref': self.ref_coords,
            'stress_gt': (stress - self.stress_mean) / self.stress_std,
            'n_nodes': self.n_nodes,
            'midx': 0,
        }


def make_visco_load_reverse_datasets(h5_path, splits_json, disp_scale=None,
                                     n_input_frames=1):
    """Create train + val datasets with shared normalization stats.

    Val uses the train split's disp_scale, stress_mean, stress_std for
    consistent normalization.
    """
    train_ds = ViscoLoadReverseDataset(
        h5_path, splits_json, split='train', disp_scale=disp_scale,
        n_input_frames=n_input_frames)
    val_ds = ViscoLoadReverseDataset(
        h5_path, splits_json, split='val',
        disp_scale=train_ds.disp_scale,
        stress_mean=train_ds.stress_mean,
        stress_std=train_ds.stress_std,
        n_input_frames=n_input_frames)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Tokenizer training mmap dataset (MicroPlate lattice)
# ---------------------------------------------------------------------------

class MmapMultiStructureDataset(Dataset):
    """Memory-mapped dataset that loads from pre-converted numpy arrays.

    Each RVE has files in mmap_dir:
        {rve_id}_coords.npy:    [N_nodes, 3] float32
        {rve_id}_outputs.npy:   [N_frames, N_nodes, 3] float32
        {rve_id}_fps_order.npy: [N_nodes] int64 (optional, for FPS subsampling)
        {rve_id}_grad_u.npy:    [N_frames, N_nodes, 3, 3] float32 (optional, Sobolev target)
        {rve_id}_P_nodes.npy:   [N_frames, N_nodes, 3, 3] float32 (optional, fallback)

    All processes mmap the same files → OS shares physical pages via page cache.
    Total RAM: ~one copy in page cache + small per-process overhead.
    """

    def __init__(self, mmap_dir, split="train", val_ratio=0.1, seed=42,
                 disp_scale=None, max_rves=0, subsample_n=0, fps_core_n=0,
                 exclude_plates=None, n_input_frames=1, n_frames_per_traj=31):
        import glob as _glob

        # Discover RVE IDs from filenames
        coord_files = sorted(_glob.glob(os.path.join(mmap_dir, '*_coords.npy')))
        rve_ids = [os.path.basename(f).replace('_coords.npy', '') for f in coord_files]
        exclude_set = set(exclude_plates or [])
        if exclude_set:
            rve_ids = [r for r in rve_ids if r not in exclude_set]
        if max_rves > 0:
            rve_ids = rve_ids[:max_rves]

        self.mesh_ids = rve_ids
        self.mesh_coords = {}  # midx → normalized coords
        self.mmap_outputs = {}  # midx → mmap'd outputs array
        self.mmap_grads = {}   # midx → mmap'd grad_u or P_nodes array (optional)
        self.fps_orders = {}   # midx → fps_order array (optional)
        self.coord_scales = {} # midx → (c_max-c_min)/2 for physical gradient scaling
        self.subsample_n = subsample_n
        self.fps_core_n = fps_core_n
        self.n_input_frames = int(n_input_frames)
        # Per-trajectory frame count for clamping multi-frame windows to a single
        # trajectory (MicroPlate lattice: 31 frames/traj, flat layout in outputs.npy).
        self.n_frames_per_traj = int(n_frames_per_traj)
        # Per-plate tet geometry for whole-plate tet Sobolev. Populated lazily
        # by attach_tet_geom(); empty dict disables tet sampling.
        self.tet_geom = {}              # midx -> dict(cells, dNdx, volumes, good, fps_tet_order)
        self.tet_k_core = 0
        self.tet_k_rand = 0
        # On-the-fly Cauchy sym6 ground truth from mmap grad_u. Filled by
        # precompute_stress_stats() or set by from_shared(); None disables.
        self.stress_mean = None
        self.stress_std = None
        self.stress_mu = MICROPLATE_LATTICE.mu
        self.stress_lmbda = MICROPLATE_LATTICE.lmbda
        self.frame_index = []  # (midx, frame_within_rve)

        t0 = time.time()
        print(f"  Loading mmap dir {mmap_dir} ({len(rve_ids)} RVEs)...")

        for midx, rve_id in enumerate(rve_ids):
            coords_path = os.path.join(mmap_dir, f'{rve_id}_coords.npy')
            outputs_path = os.path.join(mmap_dir, f'{rve_id}_outputs.npy')

            # Normalize coords to [-1, 1]
            coords = np.load(coords_path).astype(np.float32)
            c_min = coords.min(axis=0)
            c_max = coords.max(axis=0)
            scale = np.maximum(c_max - c_min, 1e-8)
            self.mesh_coords[midx] = ((coords - c_min) / scale * 2 - 1)
            self.coord_scales[midx] = (scale / 2.0).astype(np.float32)

            # Mmap outputs (no copy into RAM)
            outputs = np.load(outputs_path, mmap_mode='r')
            self.mmap_outputs[midx] = outputs
            n_frames = outputs.shape[0]

            # FPS ordering (optional, small — loaded into RAM)
            fps_path = os.path.join(mmap_dir, f'{rve_id}_fps_order.npy')
            if os.path.exists(fps_path):
                self.fps_orders[midx] = np.load(fps_path).astype(np.int64)

            # Grad_u or P_nodes for Sobolev loss (optional, mmap'd)
            # IMPORTANT: grad_u is required for Sobolev training; P_nodes is fallback for eval only
            grad_u_path = os.path.join(mmap_dir, f'{rve_id}_grad_u.npy')
            p_nodes_path = os.path.join(mmap_dir, f'{rve_id}_P_nodes.npy')
            if os.path.exists(grad_u_path):
                self.mmap_grads[midx] = np.load(grad_u_path, mmap_mode='r')
            elif os.path.exists(p_nodes_path):
                self.mmap_grads[midx] = np.load(p_nodes_path, mmap_mode='r')
                if not hasattr(self, '_p_nodes_warned'):
                    self._p_nodes_warned = True
                    print(f"  WARNING: {rve_id} uses P_nodes (not grad_u). "
                          f"Run precompute_grad_u.py + build_tokenizer_mmap.py before Sobolev training.")

            for fi in range(n_frames):
                self.frame_index.append((midx, fi))

            extras = []
            if midx in self.fps_orders:
                extras.append("fps")
            if midx in self.mmap_grads:
                extras.append("grad")
            extra_str = f" [{'+'.join(extras)}]" if extras else ""
            elapsed = time.time() - t0
            print(f"    {rve_id}: {n_frames} frames, {coords.shape[0]} nodes"
                  f"{extra_str} [{elapsed:.0f}s]")

        n_total = len(self.frame_index)
        print(f"  Total frames: {n_total}")

        # Compute displacement scale from sample
        if disp_scale is not None:
            self.disp_scale = float(disp_scale)
        else:
            rng_sample = np.random.default_rng(seed)
            n_sample = min(1000, n_total)
            sample_indices = rng_sample.choice(n_total, size=n_sample, replace=False)
            samples = []
            for si in sample_indices:
                midx, fi = self.frame_index[si]
                samples.append(np.array(self.mmap_outputs[midx][fi]).ravel())
            sampled = np.concatenate(samples)
            self.disp_scale = float(np.std(sampled)) if np.std(sampled) > 1e-8 else 1.0

        # Deterministic split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)
        split_idx = int(n_total * (1 - val_ratio))

        if split == "train":
            sel = sorted(indices[:split_idx].tolist())
        else:
            sel = sorted(indices[split_idx:].tolist())

        self.frames = [self.frame_index[i] for i in sel]

    @classmethod
    def from_shared(cls, frames, mesh_coords, mesh_ids, mmap_outputs, disp_scale,
                    mmap_grads=None, fps_orders=None, coord_scales=None,
                    subsample_n=0, fps_core_n=0,
                    n_input_frames=1, n_frames_per_traj=31,
                    tet_geom=None, tet_k_core=0, tet_k_rand=0):
        """Create dataset from pre-split data (avoids rescanning mmap dir)."""
        obj = cls.__new__(cls)
        obj.frames = frames
        obj.mesh_coords = mesh_coords
        obj.mesh_ids = mesh_ids
        obj.mmap_outputs = mmap_outputs
        obj.mmap_grads = mmap_grads or {}
        obj.fps_orders = fps_orders or {}
        obj.coord_scales = coord_scales or {}
        obj.subsample_n = subsample_n
        obj.fps_core_n = fps_core_n
        obj.n_input_frames = int(n_input_frames)
        obj.n_frames_per_traj = int(n_frames_per_traj)
        obj.tet_geom = tet_geom or {}
        obj.tet_k_core = int(tet_k_core)
        obj.tet_k_rand = int(tet_k_rand)
        obj.stress_mean = None
        obj.stress_std = None
        obj.stress_mu = MICROPLATE_LATTICE.mu
        obj.stress_lmbda = MICROPLATE_LATTICE.lmbda
        obj.disp_scale = float(disp_scale)
        return obj

    def precompute_stress_stats(self, n_samples=5000,
                                mu=MICROPLATE_LATTICE.mu,
                                lmbda=MICROPLATE_LATTICE.lmbda,
                                seed=42,
                                n_fps_core=8192):
        """Estimate per-component mean/std of Neo-Hookean Cauchy sym6 from a sample
        of mmap grad_u frames drawn from this dataset's frame index (train and
        val windows alike). Call once after construction; copy the results to
        the val split via set_stress_stats() for consistency.

        Only reads the FPS-core subset of nodes per frame (matches what the tokenizer
        sees during training) instead of the full grad_u array.

        Requires mmap_grads to be populated (i.e. grad_u precomputed for all plates).
        """
        from leia.stress import neohookean_cauchy_sym6_from_gradu
        if not self.mmap_grads:
            raise RuntimeError(
                "precompute_stress_stats requires mmap grad_u — none found.")
        self.stress_mu = float(mu)
        self.stress_lmbda = float(lmbda)
        candidates = [
            (midx, fi) for (midx, fi) in self.frame_index
            if midx in self.mmap_grads and (fi % self.n_frames_per_traj) > 0
        ]
        rng = np.random.default_rng(seed)
        n_sample = min(int(n_samples), len(candidates))
        sel_idx = rng.choice(len(candidates), size=n_sample, replace=False)
        sym6_chunks = []
        for k in sel_idx:
            midx, fi = candidates[k]
            # Read only FPS-core nodes per frame (matches training-time subsampling)
            if midx in self.fps_orders:
                idx_nodes = self.fps_orders[midx][:n_fps_core]
                grad_full = self.mmap_grads[midx][fi]   # mmap slice, lazy
                grad = np.array(grad_full[idx_nodes], dtype=np.float32)  # (n_fps_core, 3, 3)
            else:
                grad = np.array(self.mmap_grads[midx][fi], dtype=np.float32)
            sym6 = neohookean_cauchy_sym6_from_gradu(
                grad, mu=self.stress_mu, lmbda=self.stress_lmbda)         # (n, 6)
            sym6_chunks.append(sym6.reshape(-1, 6))
        all_sym6 = np.concatenate(sym6_chunks, axis=0)
        mean = all_sym6.mean(axis=0).astype(np.float32)
        std = np.maximum(all_sym6.std(axis=0), 1e-6).astype(np.float32)
        self.stress_mean = torch.tensor(mean)
        self.stress_std = torch.tensor(std)
        return self.stress_mean, self.stress_std

    def set_stress_stats(self, stress_mean, stress_std,
                         mu=MICROPLATE_LATTICE.mu,
                         lmbda=MICROPLATE_LATTICE.lmbda):
        """Copy pre-computed stats (from another split) to avoid recomputation."""
        self.stress_mean = stress_mean
        self.stress_std = stress_std
        self.stress_mu = float(mu)
        self.stress_lmbda = float(lmbda)

    def attach_tet_geom(self, tet_geom_dir, k_core, k_rand):
        """Load per-plate tet geometry from a precompute dir.

        Each plate's geometry was emitted by `scripts/precompute_tet_geom.py`
        as `{tet_geom_dir}/{plate_id}.npz` with keys:
            cells (M, 4) int64
            dNdx (M, 4, 3) float32
            volumes (M,) float32
            good (M,) bool
            fps_tet_order (K_max,) int64  — tet IDs (in original mesh) sorted by FPS

        At training time __getitem__ samples K_core deterministic tets from the
        FPS-tet ordering plus K_rand random good tets from the remainder, and
        returns a `tet_sample` dict with the union of nodes plus tet metadata
        remapped to indices into that union.
        """
        self.tet_k_core = int(k_core)
        self.tet_k_rand = int(k_rand)
        self.tet_geom = {}
        n_loaded = 0
        for midx, rve_id in enumerate(self.mesh_ids):
            path = os.path.join(tet_geom_dir, f'{rve_id}.npz')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} missing: --sobolev-method tet needs tet geometry for every "
                    f"plate in the mmap; run scripts/precompute_tet_geom.py (README § 2)")
            with np.load(path) as g:
                self.tet_geom[midx] = {
                    'cells': g['cells'].astype(np.int64),
                    'dNdx': g['dNdx'].astype(np.float32),
                    'volumes': g['volumes'].astype(np.float32),
                    'good': g['good'].astype(bool),
                    'fps_tet_order': g['fps_tet_order'].astype(np.int64),
                }
            n_loaded += 1
        print(f"  Tet geometry: loaded {n_loaded}/{len(self.mesh_ids)} plates "
              f"(K_core={k_core}, K_rand={k_rand})")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        midx, fi = self.frames[idx]
        outputs = self.mmap_outputs[midx]
        coords_norm_full = self.mesh_coords[midx]
        coords_norm = coords_norm_full

        # Current-frame displacement (reconstruction target, single frame)
        disp_t_full = np.array(outputs[fi], dtype=np.float32) / self.disp_scale
        disp_t = disp_t_full

        # Multi-frame window for tokenizer input. Clamp to the current trajectory
        # (frame 0 of each trajectory is rest state, so crossing boundaries would
        # inject a spurious displacement discontinuity).
        if self.n_input_frames == 3:
            local_t = fi % self.n_frames_per_traj
            traj_start = fi - local_t
            fi_m2 = traj_start + max(local_t - 2, 0)
            fi_m1 = traj_start + max(local_t - 1, 0)
            disp_m2_full = np.array(outputs[fi_m2], dtype=np.float32) / self.disp_scale
            disp_m1_full = np.array(outputs[fi_m1], dtype=np.float32) / self.disp_scale
            disp_m2 = disp_m2_full
            disp_m1 = disp_m1_full
        else:
            disp_m2 = None
            disp_m1 = None
            disp_m2_full = None
            disp_m1_full = None

        # Grad/stress target (optional)
        grad_frame_full = None
        grad_frame = None
        if midx in self.mmap_grads:
            grad_frame_full = np.array(self.mmap_grads[midx][fi], dtype=np.float32)
            grad_frame = grad_frame_full

        # FPS subsampling: deterministic core + a remainder drawn with a fixed seed (same nodes per plate every epoch)
        selected = None
        if self.subsample_n > 0 and midx in self.fps_orders:
            N = coords_norm.shape[0]
            if N > self.subsample_n:
                from leia.tokenizer import subsample_fps
                selected = subsample_fps(self.fps_orders[midx],
                                         self.subsample_n, self.fps_core_n)

                disp_t = disp_t[selected]
                coords_norm = coords_norm[selected]
                if disp_m2 is not None:
                    disp_m2 = disp_m2[selected]
                    disp_m1 = disp_m1[selected]
                if grad_frame is not None:
                    grad_frame = grad_frame[selected]

        # Whole-plate tet Sobolev sampling: pick K_core deterministic + K_rand
        # random good tets, gather union of their 4 nodes (in full-mesh
        # indices), then de-duplicate against the FPS sample. Extra (non-FPS)
        # nodes are appended to ref/grad_u so the decoder can be queried at
        # them; the recon loss reads the FPS prefix, the tet Sobolev loss reads
        # the tet-only suffix and uses cells_remapped indices into the union.
        tet_sample = None
        if (self.tet_geom and midx in self.tet_geom
                and (self.tet_k_core + self.tet_k_rand) > 0
                and grad_frame_full is not None):
            tg = self.tet_geom[midx]
            cells_full = tg['cells']           # (M, 4) int64
            dNdx_full = tg['dNdx']             # (M, 4, 3) float32
            volumes_full = tg['volumes']       # (M,) float32
            good_full = tg['good']             # (M,) bool
            fps_tet_order = tg['fps_tet_order']  # (K_max,) int64 — good tet IDs

            k_core = min(self.tet_k_core, len(fps_tet_order))
            core_tet_ids = fps_tet_order[:k_core] if k_core > 0 else np.empty(0, dtype=np.int64)

            # Random remainder: uniform over good tets NOT already in core.
            if self.tet_k_rand > 0:
                good_idx = np.flatnonzero(good_full)
                # Mask out the core tet IDs from the sampling pool.
                core_set = set(core_tet_ids.tolist())
                pool = np.array([t for t in good_idx if t not in core_set],
                                dtype=np.int64)
                k_rand = min(self.tet_k_rand, pool.size)
                rand_choice = np.random.choice(pool, size=k_rand, replace=False)
            else:
                rand_choice = np.empty(0, dtype=np.int64)

            tet_ids = np.concatenate([core_tet_ids, rand_choice]).astype(np.int64)
            sampled_cells = cells_full[tet_ids]   # (K, 4) int64
            sampled_dNdx = dNdx_full[tet_ids]     # (K, 4, 3) float32
            sampled_volumes = volumes_full[tet_ids]  # (K,) float32

            # Full-mesh node IDs needed for these tets.
            tet_node_ids = np.unique(sampled_cells.reshape(-1)).astype(np.int64)

            # Identify which tet nodes are NOT in the existing FPS sample.
            if selected is not None:
                fps_set = set(selected.tolist())
                extra_mask = np.array([nid not in fps_set for nid in tet_node_ids],
                                       dtype=bool)
            else:
                # No FPS subsampling: union is full mesh, so no extras needed.
                # We still need a remap to local indices for `cells_remapped`,
                # but `selected` is `arange(N_full)` effectively.
                extra_mask = np.zeros(tet_node_ids.shape, dtype=bool)

            extra_node_ids = tet_node_ids[extra_mask]   # (N_extra,) int64

            # Build the FPS+extras union (if FPS was applied; else full mesh).
            if selected is not None:
                union_indices = np.concatenate([selected, extra_node_ids]).astype(np.int64)
            else:
                union_indices = np.arange(coords_norm.shape[0], dtype=np.int64)

            # Append extras to the decoder query / target tensors.
            if extra_node_ids.size > 0 and selected is not None:
                coords_extra = coords_norm_full[extra_node_ids]
                disp_extra = disp_t_full[extra_node_ids]
                coords_norm = np.concatenate([coords_norm, coords_extra], axis=0)
                disp_t = np.concatenate([disp_t, disp_extra], axis=0)
                if disp_m2 is not None:
                    disp_m2 = np.concatenate([disp_m2, disp_m2_full[extra_node_ids]], axis=0)
                    disp_m1 = np.concatenate([disp_m1, disp_m1_full[extra_node_ids]], axis=0)
                if grad_frame is not None:
                    grad_frame_extra = grad_frame_full[extra_node_ids]
                    grad_frame = np.concatenate([grad_frame, grad_frame_extra], axis=0)

            # Remap cells from full-mesh indices to indices within the union.
            # union_indices[k] = full-mesh node ID at union slot k. Build the
            # inverse mapping: full_to_union[full_id] = union slot.
            full_to_union = np.full(coords_norm_full.shape[0], -1, dtype=np.int64)
            full_to_union[union_indices] = np.arange(len(union_indices), dtype=np.int64)
            cells_remapped = full_to_union[sampled_cells]   # (K, 4) int64
            assert (cells_remapped >= 0).all(), "tet_sample remap had missing nodes"

            n_fps = len(selected) if selected is not None else coords_norm_full.shape[0]
            tet_sample = {
                'cells_remapped': torch.tensor(cells_remapped, dtype=torch.long),
                'dNdx': torch.tensor(sampled_dNdx, dtype=torch.float32),
                'volumes': torch.tensor(sampled_volumes, dtype=torch.float32),
                'n_fps': int(n_fps),
                'n_extra': int(extra_node_ids.size),
            }

        if self.n_input_frames == 3:
            x_input = np.concatenate([disp_m2, disp_m1, disp_t], axis=-1)  # (N, 9)
        else:
            x_input = disp_t

        result = {
            'x': torch.tensor(x_input, dtype=torch.float32),
            'x_target': torch.tensor(disp_t, dtype=torch.float32),
            'ref': torch.tensor(coords_norm, dtype=torch.float32),
            'n_nodes': disp_t.shape[0],
            'midx': midx,
        }
        if tet_sample is not None:
            result['tet_sample'] = tet_sample

        if grad_frame is not None:
            result['P_gt'] = torch.tensor(grad_frame, dtype=torch.float32)

            # On-the-fly Neo-Hookean Cauchy sym6 ground truth for stress head.
            # grad_u in mmap is already in physical units, so F = I + grad_u
            # directly yields the correct deformation gradient.
            if self.stress_mean is not None and self.stress_std is not None:
                from leia.stress import neohookean_cauchy_sym6_from_gradu
                sym6 = neohookean_cauchy_sym6_from_gradu(
                    grad_frame, mu=self.stress_mu, lmbda=self.stress_lmbda)  # (N, 6)
                sym6_t = torch.tensor(sym6, dtype=torch.float32)
                normalized = (sym6_t - self.stress_mean) / self.stress_std
                result['stress_gt'] = normalized

        if midx in self.coord_scales:
            result['coord_scale'] = torch.tensor(
                self.coord_scales[midx], dtype=torch.float32)

        return result






def padding_collate(batch):
    """Pad variable-size meshes to the largest in the batch.

    Auto-detects the tokenizer input feature dimension from batch[0]['x'] (3 for
    single-frame, 9 for n_input_frames=3), and propagates optional x_target
    (shape [N, 3]) and stress_gt (shape [N, 6]) through the same padding.
    """
    max_v = max(item['x'].shape[0] for item in batch)
    B = len(batch)
    x_feat = batch[0]['x'].shape[-1]  # 3 for single-frame, 9 for n_input_frames=3

    x_padded = torch.zeros(B, max_v, x_feat)
    ref_padded = torch.zeros(B, max_v, 3)
    padding_mask = torch.ones(B, max_v, dtype=torch.bool)
    midx = torch.tensor([item['midx'] for item in batch], dtype=torch.long)

    has_target = 'x_target' in batch[0] and batch[0]['x_target'] is not None
    x_target_padded = (
        torch.zeros(B, max_v, batch[0]['x_target'].shape[-1]) if has_target else None)

    has_stress = 'stress_gt' in batch[0] and batch[0]['stress_gt'] is not None
    stress_gt_padded = (
        torch.zeros(B, max_v, batch[0]['stress_gt'].shape[-1]) if has_stress else None)

    for i, item in enumerate(batch):
        v = item['x'].shape[0]
        x_padded[i, :v] = item['x']
        ref_padded[i, :v] = item['ref']
        padding_mask[i, :v] = False
        if has_target:
            x_target_padded[i, :v] = item['x_target']
        if has_stress:
            stress_gt_padded[i, :v] = item['stress_gt']

    # Optional stress fields (from FPS-subsampled lattice data)
    P_gt = None
    coord_scale = None
    if 'P_gt' in batch[0] and batch[0]['P_gt'] is not None:
        P_gt_padded = torch.zeros(B, max_v, 3, 3)
        for i, item in enumerate(batch):
            v = item['P_gt'].shape[0]
            P_gt_padded[i, :v] = item['P_gt']
        P_gt = P_gt_padded
    if 'coord_scale' in batch[0] and batch[0]['coord_scale'] is not None:
        coord_scale = torch.stack([item['coord_scale'] for item in batch])

    # Optional tet-Sobolev sample bundle (whole-plate tet shape function path).
    # K is fixed across samples in a batch since (K_core+K_rand) is per-dataset.
    # cells_remapped indices point into each sample's [FPS ∪ extras] union;
    # those indices are < n_fps + n_extra, so they're all in the unpadded
    # region of x/ref/P_gt and don't need any further remapping.
    tet_sample = None
    if 'tet_sample' in batch[0] and batch[0]['tet_sample'] is not None:
        ts0 = batch[0]['tet_sample']
        K = ts0['cells_remapped'].shape[0]
        cells_remapped = torch.zeros(B, K, 4, dtype=torch.long)
        dNdx_b = torch.zeros(B, K, 4, 3, dtype=torch.float32)
        volumes_b = torch.zeros(B, K, dtype=torch.float32)
        n_fps_arr = torch.zeros(B, dtype=torch.long)
        n_extra_arr = torch.zeros(B, dtype=torch.long)
        for i, item in enumerate(batch):
            ts = item['tet_sample']
            cells_remapped[i] = ts['cells_remapped']
            dNdx_b[i] = ts['dNdx']
            volumes_b[i] = ts['volumes']
            n_fps_arr[i] = ts['n_fps']
            n_extra_arr[i] = ts['n_extra']
        tet_sample = {
            'cells_remapped': cells_remapped,
            'dNdx': dNdx_b,
            'volumes': volumes_b,
            'n_fps': n_fps_arr,
            'n_extra': n_extra_arr,
        }

    return {
        'x': x_padded,
        'x_target': x_target_padded if has_target else x_padded,  # fallback to x for single-frame
        'ref': ref_padded,
        'padding_mask': padding_mask,
        'midx': midx,
        'P_gt': P_gt,
        'stress_gt': stress_gt_padded,
        'coord_scale': coord_scale,
        'tet_sample': tet_sample,
    }

