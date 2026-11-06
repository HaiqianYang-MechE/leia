"""Pure metric primitives — arrays in, numbers out.

No model, no I/O. Used by the eval pipelines and any caller that needs the
exact same correlation / error definitions.
"""

import numpy as np


def pearson_corr(a, b):
    """Pearson correlation coefficient between two 1-D arrays.

    Returns 0.0 when undefined (zero variance, length < 2).
    """
    if len(a) < 2:
        return 0.0
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom < 1e-30:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def disp_rel_error(pred, gt, eps=1e-30):
    """Relative L2 displacement error: ||pred - gt|| / max(||gt||, eps)."""
    return float(np.linalg.norm(pred - gt) / max(np.linalg.norm(gt), eps))


def displacement_relative_l2(pred, gt, eps=1e-10):
    """Per-frame relative L2 over a batch of trajectories.

    Args:
        pred, gt : (..., T, N, 3) — same shape; T frames, N nodes, 3D disp.
        eps : guard against zero-displacement frames.

    Returns:
        rel_l2 : (..., T) — per-frame relative L2 = ||pred-gt||_F / ||gt||_F.
    """
    diff_norm = np.sqrt(((pred - gt) ** 2).sum(axis=-1).sum(axis=-1))
    gt_norm = np.sqrt((gt ** 2).sum(axis=-1).sum(axis=-1))
    gt_norm = np.maximum(gt_norm, eps)
    return diff_norm / gt_norm


def vm_corr(pred_vm, gt_vm, mask=None):
    """Pearson correlation of per-element von Mises stress.

    Args:
        pred_vm: [N] predicted VM stress per element.
        gt_vm: [N] ground-truth VM stress per element.
        mask: optional [N] bool, restrict to selected elements (e.g. valid tets).
    """
    if mask is not None:
        pred_vm = pred_vm[mask]
        gt_vm = gt_vm[mask]
    return pearson_corr(pred_vm, gt_vm)


def vm_from_sym6(sym6):
    """Cauchy sym6 (..., 6) → von Mises (...,).

    Voigt order is (xx, yy, zz, xy, xz, yz). Works on numpy arrays or torch
    tensors uniformly via duck-typed indexing + sqrt.
    """
    xx, yy, zz, xy, xz, yz = (sym6[..., i] for i in range(6))
    tr = xx + yy + zz
    sxx, syy, szz = xx - tr / 3.0, yy - tr / 3.0, zz - tr / 3.0
    s_sq = (sxx * sxx + syy * syy + szz * szz
            + 2.0 * (xy * xy + xz * xz + yz * yz))
    if hasattr(sym6, 'sqrt'):  # torch.Tensor
        return (1.5 * s_sq.clamp(min=0.0)).sqrt()
    return np.sqrt(1.5 * np.maximum(s_sq, 0.0))


def fd_curve_metrics(pred_force, gt_force, pred_disp=None, gt_disp=None):
    """Metrics for force-displacement curve comparison.

    Returns a dict with relative L2 of the force curve, and optionally of the
    displacement curve.
    """
    out = {
        'force_rel_l2': disp_rel_error(np.asarray(pred_force), np.asarray(gt_force)),
    }
    if pred_disp is not None and gt_disp is not None:
        out['disp_rel_l2'] = disp_rel_error(np.asarray(pred_disp), np.asarray(gt_disp))
    return out
