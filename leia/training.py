"""Training-loop primitives: checkpoint I/O + AR rollout eval.

Functions, NOT a Trainer class. Composed by `scripts/train_tokenizer.py` and
`scripts/train_dynamics.py` via direct calls; importable from notebooks for
ad-hoc continuation runs.

Only the genuinely-reusable plumbing lives here (checkpoint I/O, the canonical
save-dict shape, latent-space AR rollout eval). The training math itself
(forward + loss composition + pushforward rollout) stays inline in the scripts
— the math has script-specific shape and trying to extract a generic
train_one_epoch invites regressions.
"""

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def save_checkpoint(path: str, model: torch.nn.Module, *,
                    epoch: int, config: Dict,
                    optimizer=None, scheduler=None,
                    extras: Optional[Dict] = None):
    """Save a LEIA checkpoint with the canonical layout.

    Top-level keys: epoch, config, model_state_dict, plus optional
    optimizer_state_dict / scheduler_state_dict if those args are passed,
    plus any auxiliary numerics merged from `extras` (disp_scale,
    stress_mean, stress_std, val_loss, ar_rollout_mse, global_step, ...).

    Strips DDP wrapping so the saved state_dict matches the bare-model layout
    that `from_checkpoint()` classmethods expect. Creates parent directories.
    """
    raw = model.module if hasattr(model, 'module') else model

    payload = {
        'epoch': epoch,
        'config': config,
        'model_state_dict': raw.state_dict(),
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler_state_dict'] = scheduler.state_dict()
    if extras:
        payload.update(extras)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Tuple[Dict, Dict, Dict]:
    """Load a LEIA checkpoint.

    Returns
    -------
    (state_dict, optimizer_state, payload) where:
        state_dict: model_state_dict ready for `model.load_state_dict(...)`
        optimizer_state: optimizer_state_dict if present, else None
        payload: the full ckpt dict (so callers can read epoch, disp_scale,
                 stress_mean, stress_std, etc. directly)

    Always uses `weights_only=False` — these checkpoints carry numpy arrays
    for normalization stats which don't survive the strict unpickle path.
    """
    payload = torch.load(path, map_location=map_location or 'cpu',
                         weights_only=False)
    state_dict = payload.get('model_state_dict', payload)
    optimizer_state = payload.get('optimizer_state_dict')
    return state_dict, optimizer_state, payload


def resume_training_state(path, model, optimizer, scheduler, *,
                          lr_schedule, reset_optimizer=False,
                          map_location=None, steps_per_epoch=None,
                          verbose=True):
    """Resume model + optimizer + scheduler from a LEIA checkpoint.

    Restores the model state (strict=False so legacy keys are ignored).
    Optimizer/scheduler restoration is skipped when `reset_optimizer=True`
    OR when `lr_schedule` mismatches the saved config (e.g. cosine→flat —
    the saved scheduler state would replay the wrong schedule).

    Args:
        path: checkpoint path.
        model: bare or DDP-wrapped model; DDP wrapping is handled.
        optimizer: torch optimizer to restore into.
        scheduler: torch LR scheduler to restore into; `last_epoch` is
            advanced to the resumed `global_step` when no scheduler_state
            is present in the ckpt.
        lr_schedule: the *current run's* lr schedule, compared against
            the ckpt's saved config.
        reset_optimizer: force fresh optimizer regardless of schedule match.
        map_location: passed to torch.load.
        steps_per_epoch: only used as a fallback to estimate `global_step`
            when an old checkpoint was saved without it.
        verbose: whether to print progress (typically `is_main_rank()`).

    Returns:
        dict with keys: start_epoch, global_step, best_val_loss,
        best_ar_mse (dynamics), best_val_recon (tokenizer), fresh_optimizer.
        The two "best" selectors are restored so that the first validation
        after a resume cannot overwrite best_ar.pt / best_recon.pt with a
        worse value.
    """
    state_dict, opt_state, ckpt = load_checkpoint(path, map_location=map_location)
    raw_model = model.module if hasattr(model, 'module') else model
    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
    if verbose and (missing or unexpected):
        print(f"  Checkpoint compat: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")

    ckpt_lr_schedule = ckpt.get('config', {}).get('lr_schedule', 'cosine')
    fresh_optimizer = reset_optimizer or (lr_schedule != ckpt_lr_schedule)

    out = {
        'start_epoch': 0,
        'global_step': 0,
        'best_val_loss': float('inf'),
        'best_ar_mse': float('inf'),
        'best_val_recon': float('inf'),
        'fresh_optimizer': fresh_optimizer,
    }

    # The best-so-far selectors are independent of the optimizer: restore them
    # on every resume so best.pt / best_ar.pt / best_recon.pt are never
    # overwritten by a worse value after the restart.
    out['best_val_loss'] = ckpt.get('best_val_loss', float('inf'))
    out['best_ar_mse'] = ckpt.get('best_ar_mse', float('inf'))
    out['best_val_recon'] = ckpt.get('best_val_recon', float('inf'))

    if fresh_optimizer:
        if verbose:
            reason = ('--reset-optimizer' if reset_optimizer
                      else f'lr_schedule changed ({ckpt_lr_schedule}→{lr_schedule})')
            print(f"  Fresh optimizer ({reason}): skipping optimizer/scheduler "
                  f"state, resetting counters")
        return out

    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
        out['start_epoch'] = ckpt.get('epoch', -1) + 1

    if steps_per_epoch is not None:
        out['global_step'] = ckpt.get('global_step', out['start_epoch'] * steps_per_epoch)
    else:
        out['global_step'] = ckpt.get('global_step', 0)

    sched_state = ckpt.get('scheduler_state_dict')
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)
    else:
        scheduler.last_epoch = out['global_step']

    if verbose:
        print(f"Resumed from epoch {out['start_epoch']}, "
              f"step {out['global_step']}, "
              f"best_val_loss={out['best_val_loss']:.6f}")
    return out


@torch.no_grad()
def ar_rollout_eval(model, ar_gt, ar_actions, velocity_input):
    """Latent-space AR rollout eval for a dynamics model.

    Rolls out the dynamics model autoregressively from the GT rest frame and
    compares predicted embeddings with GT at each step, with the same pairing
    the model is trained on and that `eval_e2e` uses: from frame t, action t
    predicts frame t+1, and the rollout starts at frame 0 with zero velocity
    (prev_state = state). Runs entirely in latent space — no tokenizer decode,
    so very fast (~2 s for 5 trajectories).

    Args:
        model: dynamics model (raw, not DDP-wrapped), in eval mode.
        ar_gt: [n_traj, n_frames, K, H] GT embeddings, frame 0 = rest state.
        ar_actions: [n_traj, n_frames - 1, action_dim] actions, already
            action-scaled; action t drives frame t -> t+1.
        velocity_input: bool — whether the model expects prev_state.

    Returns:
        mean_mse: scalar, average MSE over all rollout frames and trajectories.
        per_frame_mse: np.ndarray of shape [n_frames - 1] — average MSE at
            each rollout step.
        elapsed: wall time in seconds.
    """
    t0 = time.time()
    n_traj, n_frames, K, H = ar_gt.shape

    state = ar_gt[:, 0]        # z(0), the rest frame
    prev_state = state         # zero velocity at t = 0

    frame_mses = []
    for t in range(n_frames - 1):
        action = ar_actions[:, t]
        pred = model(state, action,
                     prev_state=prev_state if velocity_input else None)
        gt_next = ar_gt[:, t + 1]
        mse = (pred - gt_next).pow(2).mean(dim=(1, 2))  # [n_traj]
        frame_mses.append(mse.mean().item())
        prev_state = state
        state = pred

    elapsed = time.time() - t0
    per_frame = np.array(frame_mses)
    return float(per_frame.mean()), per_frame, elapsed
