#!/usr/bin/env python
"""
Train the AdaLNContinuousDynamicsTransformer.

Consumes a pre-encoded latent mmap (`{plate_id}_prevq.npy`,
`{plate_id}_cumbc.npy`) produced by `scripts/build_dynamics_mmap.py` and
optimizes continuous next-state regression with pushforward
(`--method continuous_pf`, `--rollout-steps N`). Always launch via
`scripts/launch.sh`.

Usage:
    bash scripts/launch.sh train-dynamics --gpus 8 --run-name v2-dyn -- \
        --tokenizer-checkpoint checkpoints/tokenizer/v2-tok_*/best.pt \
        --mmap-dir data/microplate_lattice/structure_tokens_prod_mmap \
        --method continuous_pf --rollout-steps 10 --velocity-input \
        --noise-std 0.02 --dropout 0.3 --batch-size 8 \
        --lr 1e-4 --epochs 300 --warmup-epochs 10 --lr-schedule cosine \
        --bf16 --ar-eval-trajs 5 --ar-eval-interval 1 --wandb
"""

import os
import sys
import time
import math
import argparse
import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.dynamics import (
    AdaLNContinuousDynamicsTransformer,
    DYNAMICS_HEADS, DYNAMICS_LAYERS, DYNAMICS_DROPOUT,
)
from leia.datasets import MmapWholePlateTrajectoryDataset
from leia.regime import MICROPLATE_LATTICE
from leia.config import DYNAMICS_CHECKPOINT_ROOT, WANDB_PROJECT_DYNAMICS
from leia.distributed import setup_ddp, teardown_ddp, is_main_rank
from leia.training import (save_checkpoint, load_checkpoint, ar_rollout_eval,
                            resume_training_state)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train DynamicsTransformer")
    parser.add_argument('--mmap-dir', type=str, required=True,
                        help='Mmap directory of whole-plate dynamics data')
    parser.add_argument('--mmap-dir-val', type=str, default=None,
                        help='Separate mmap directory for validation '
                             '(if omitted, a 10%% slice of --mmap-dir is held out)')
    parser.add_argument('--exclude-plates', type=str, default='',
                        help='Comma-separated plate IDs to exclude from training')
    parser.add_argument('--tokenizer-checkpoint', type=str, required=True,
                        help='Trained tokenizer checkpoint (.pt). Used to read '
                             'embed_dim / num_latents and to record provenance.')
    parser.add_argument('--run-name', type=str, required=True,
                        help='Run name (used for checkpoint dir + W&B run)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Per-GPU batch size (production lattice: 8 × 8 GPU = 64)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--checkpoint-interval', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='When resuming, skip optimizer/scheduler state '
                             '(use this to switch lr_schedule mid-run).')
    parser.add_argument('--action-scale', type=float, default=1.0,
                        help='Multiply action by this factor at training time')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of epochs for linear LR warmup (0=use --warmup-steps)')
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='LR warmup in optimizer steps (0=use --warmup-epochs)')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'flat'],
                        help='LR schedule after warmup (cosine=production)')
    parser.add_argument('--max-groups', type=int, default=0,
                        help='Limit dataset to first N plates (0=all; debug only)')
    parser.add_argument('--method', type=str, default='continuous_pf',
                        choices=['continuous_pf'],
                        help='Training method: continuous_pf, a multi-step '
                             'differentiable rollout (production). Single-step '
                             'training is --rollout-steps 1.')
    parser.add_argument('--rollout-steps', type=int, default=10,
                        help='Number of rollout steps for continuous_pf '
                             '(production lattice: 10; visco-LR: 20)')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='Use bfloat16 mixed precision (default: on)')
    parser.add_argument('--no-bf16', action='store_false', dest='bf16',
                        help='Disable bfloat16 (use fp32)')
    parser.add_argument('--residual', action='store_true',
                        help='Delta prediction: output = state + head(x)')
    parser.add_argument('--noise-std', type=float, default=0.02,
                        help='Gaussian noise std on input latents during training '
                             '(production lattice: 0.02; visco-LR: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--velocity-input', action='store_true',
                        help='Concatenate latent velocity (z_t - z_{t-1}) with state input')
    parser.add_argument('--diff-steps', type=int, default=0,
                        help='Differentiable rollout steps in continuous_pf '
                             '(0 = all steps differentiable)')
    parser.add_argument('--ar-eval-trajs', type=int, default=5,
                        help='Number of trajectories for latent AR rollout eval. '
                             '0 disables; the production checkpoint selection '
                             '(`best_ar.pt`) requires this > 0.')
    parser.add_argument('--ar-eval-interval', type=int, default=1,
                        help='Run AR rollout eval every N epochs')
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()
    is_main = is_main_rank()

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Validate diff-steps
    if args.diff_steps > 0 and args.diff_steps > args.rollout_steps:
        if is_main:
            print(f"Warning: --diff-steps {args.diff_steps} > --rollout-steps "
                  f"{args.rollout_steps}, clamping to rollout-steps")
        args.diff_steps = args.rollout_steps

    if is_main:
        print(f"Device: {device}, World size: {world_size}")
        print(f"Data: {args.mmap_dir}")
        print(f"Method: {args.method}")
        if args.method == 'continuous_pf':
            diff = args.diff_steps if args.diff_steps > 0 else args.rollout_steps
            print(f"Pushforward: {diff} diff + "
                  f"{args.rollout_steps - diff} detached = "
                  f"{args.rollout_steps} total steps")

    # --- Dataset ---
    exclude_plates = (
        [p.strip() for p in args.exclude_plates.split(',') if p.strip()]
        if args.exclude_plates else None)

    split_seed = MICROPLATE_LATTICE.splits.random_split_seed
    split_ratio = MICROPLATE_LATTICE.splits.random_val_ratio
    val_ratio_train = 0.0 if args.mmap_dir_val else split_ratio
    train_dataset = MmapWholePlateTrajectoryDataset(
        args.mmap_dir, split='train',
        val_ratio=val_ratio_train,
        seed=split_seed,
        action_scale=args.action_scale,
        max_groups=args.max_groups,
        rollout_steps=args.rollout_steps,
        exclude_plates=exclude_plates)
    val_mmap = args.mmap_dir_val or args.mmap_dir
    val_dataset = MmapWholePlateTrajectoryDataset(
        val_mmap, split='train' if args.mmap_dir_val else 'val',
        val_ratio=0.0 if args.mmap_dir_val else split_ratio,
        seed=split_seed,
        action_scale=args.action_scale,
        max_groups=args.max_groups,
        rollout_steps=args.rollout_steps,
        exclude_plates=exclude_plates)
    action_dim = train_dataset.action_dim
    if is_main:
        print(f"Whole-plate dynamics: action_dim={action_dim}, "
              f"action_scale={args.action_scale}")
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=3,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=3,
    )

    # --- Model ---
    # Read embed_dim / num_latents from the tokenizer checkpoint config.
    tok_ckpt = torch.load(args.tokenizer_checkpoint, map_location='cpu', weights_only=False)
    tok_cfg = tok_ckpt.get('config', {})
    tok_embed_dim = tok_cfg['hidden_dim']
    tok_num_latents = tok_cfg['num_latents']
    del tok_ckpt
    if is_main:
        print(f"Tokenizer config: embed_dim={tok_embed_dim}, num_latents={tok_num_latents}")

    model = AdaLNContinuousDynamicsTransformer(
        num_latents=tok_num_latents,
        embed_dim=tok_embed_dim,
        action_dim=action_dim,
        heads=DYNAMICS_HEADS,
        layers=DYNAMICS_LAYERS,
        dropout=args.dropout,
        residual=args.residual,
        velocity_input=args.velocity_input,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model params: {n_params/1e6:.2f}M")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        fused=True,
    )

    # Step-level warmup + cosine decay. --warmup-steps wins over
    # --warmup-epochs when both are set.
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if args.lr_schedule == 'flat':
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    if args.resume:
        rs = resume_training_state(
            args.resume, model, optimizer, scheduler,
            lr_schedule=args.lr_schedule,
            reset_optimizer=args.reset_optimizer,
            map_location=device,
            steps_per_epoch=steps_per_epoch,
            verbose=is_main)
        start_epoch = rs['start_epoch']
        best_val_loss = rs['best_val_loss']
        best_ar_mse = rs['best_ar_mse']      # so a resume cannot overwrite best_ar.pt with a worse epoch
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        best_ar_mse = float('inf')

    # Checkpoint dir
    checkpoint_dir = os.path.join(
        DYNAMICS_CHECKPOINT_ROOT,
        f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints: {checkpoint_dir}")

    # W&B
    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT_DYNAMICS,
            name=args.run_name,
            config=vars(args),
        )

    def _save_dyn(filename, *, optimizer=None, scheduler=None, **extras):
        """Save a dynamics checkpoint with the standard metadata fields.

        Closure over `model`, `args`, `action_dim`, `checkpoint_dir`, `is_main`.
        Caller passes per-checkpoint extras (val_loss, ar_rollout_mse, etc.)
        and optional optimizer/scheduler state for periodic snapshots.
        """
        if not is_main:
            return
        base = {
            'method': args.method,
            'action_dim': action_dim,
            'action_scale': args.action_scale,
        }
        base.update(extras)
        save_checkpoint(
            os.path.join(checkpoint_dir, filename), model,
            epoch=epoch, config=vars(args),
            optimizer=optimizer, scheduler=scheduler, extras=base)

    # Load GT embeddings for AR rollout eval (rank 0 only)
    ar_eval_data = None
    ar_actions_eval = None
    if is_main and args.ar_eval_trajs > 0 and args.mmap_dir:
        n_ar = args.ar_eval_trajs
        prevq_map = train_dataset._prevq_maps[0]
        n_frames_ar = train_dataset._n_frames
        n_traj_total = prevq_map.shape[0] // n_frames_ar
        traj_start = n_traj_total - n_ar  # use last N trajectories
        ar_gt_list = []
        for ti in range(n_ar):
            idx = (traj_start + ti) * n_frames_ar
            ar_gt_list.append(torch.tensor(
                np.array(prevq_map[idx:idx + n_frames_ar]),
                dtype=torch.float32, device=device))
        ar_eval_data = torch.stack(ar_gt_list)  # [n_ar, n_frames, K, H]
        ar_actions_eval = torch.tensor(
            train_dataset._cumbc_arrays[0][traj_start:traj_start + n_ar]
            * args.action_scale,
            dtype=torch.float32, device=device)
        print(f"AR eval: {n_ar} trajectories, {n_frames_ar} frames, "
              f"traj indices {traj_start}..{traj_start + n_ar - 1}")

    # ===================================================================
    # Training loop
    # bf16 autocast setup (no GradScaler — bf16 has sufficient dynamic range)
    use_bf16 = args.bf16 and torch.cuda.is_available()
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext()
    if is_main and use_bf16:
        print(f"Using bfloat16 mixed precision")

    # ===================================================================
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # =============================================================
            # METHOD: CONTINUOUS_PF — differentiable multi-step rollout
            # =============================================================
            rollout_steps = args.rollout_steps
            diff_steps = args.diff_steps if args.diff_steps > 0 else rollout_steps

            # Mmap trajectory format: prevq [B, k+2, K, H], actions [B, k, A]
            # prevq[:, 0] = z(t-1), prevq[:, 1] = z(t), prevq[:, 2:] = targets
            prevq_embs = batch['prevq'].to(device)
            actions = batch['actions'].to(device)
            prev_state = prevq_embs[:, 0]   # z(t-1) for velocity
            state_emb = prevq_embs[:, 1]    # z(t) current state
            target_offset = 2               # targets start at index 2

            step_loss = torch.tensor(0.0, device=device)

            # Phase 1: Differentiable steps (k=0..diff_steps-1)
            for k in range(min(diff_steps, rollout_steps)):
                action = actions[:, k]
                target_emb = prevq_embs[:, k + target_offset]

                if args.noise_std > 0 and model.training:
                    state_emb = state_emb + args.noise_std * torch.randn_like(state_emb)

                with autocast_ctx:
                    pred_emb = model(state_emb, action,
                                     prev_state=prev_state if args.velocity_input else None)
                    step_loss = step_loss + F.mse_loss(pred_emb, target_emb)

                prev_state = state_emb
                state_emb = pred_emb

            # Phase 2: Detached steps (k=diff_steps..rollout_steps-1)
            if diff_steps < rollout_steps:
                with torch.no_grad():
                    for k in range(diff_steps, rollout_steps):
                        action = actions[:, k]

                        with autocast_ctx:
                            pred_emb = model(
                                state_emb, action,
                                prev_state=prev_state if args.velocity_input else None)

                        prev_state = state_emb
                        state_emb = pred_emb

            # MSE loss on differentiable steps only
            loss = step_loss / min(diff_steps, rollout_steps)
            loss.backward()

            # --- Optimizer step + accumulate metrics ---
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if wandb_run and batch_idx % args.log_interval == 0:
                wandb_run.log({'train/loss_step': loss.item()})

        # =============================================================
        # Validation (single-step MSE — first frame of each trajectory)
        # =============================================================
        model.eval()
        val_loss = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                # Trajectory format: prevq layout [z(t-1), z(t), z(t+1), ..., z(t+k)]
                action = batch['actions'][:, 0].to(device)
                state_in_emb = batch['prevq'][:, 1].to(device)   # z(t)
                state_out_emb = batch['prevq'][:, 2].to(device)  # z(t+1)
                prev_state_val = (batch['prevq'][:, 0].to(device)
                                  if args.velocity_input else None)

                with autocast_ctx:
                    pred_emb = model(state_in_emb, action, prev_state=prev_state_val)
                    loss = F.mse_loss(pred_emb, state_out_emb)

                val_loss += loss.item()
                val_n += 1

        # Aggregate validation metrics across DDP ranks
        if world_size > 1:
            val_tensors = torch.tensor(
                [val_loss, val_n], dtype=torch.float64, device=device)
            dist.all_reduce(val_tensors)
            val_loss, val_n = val_tensors.tolist()

        # Epoch summary
        avg_train_loss = total_loss / max(n_batches, 1)
        avg_val_loss = val_loss / max(val_n, 1)
        dt = time.time() - t0

        if is_main:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | "
                  f"train loss={avg_train_loss:.4f} | "
                  f"val loss={avg_val_loss:.4f} | "
                  f"lr={lr:.2e} | {dt:.1f}s")

            if wandb_run:
                wandb_run.log({
                    'epoch': epoch,
                    'train/loss': avg_train_loss,
                    'val/loss': avg_val_loss,
                    'lr': lr,
                })

            # Best checkpoint by val_loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                _save_dyn('best.pt',
                          val_loss=avg_val_loss, best_val_loss=best_val_loss,
                          best_ar_mse=best_ar_mse)
                print(f"  New best val_loss={best_val_loss:.4f}")

            # AR rollout eval (latent-space, no decode — very fast)
            ar_rollout_mse = None
            if (ar_eval_data is not None
                    and (epoch + 1) % args.ar_eval_interval == 0):
                raw_model = model.module if hasattr(model, 'module') else model
                raw_model.eval()
                ar_rollout_mse, ar_per_frame, ar_time = ar_rollout_eval(
                    raw_model, ar_eval_data, ar_actions_eval,
                    args.velocity_input)
                ar_late_mse = (float(ar_per_frame[500:].mean())
                               if len(ar_per_frame) > 500
                               else ar_rollout_mse)
                print(f"  AR rollout: mse={ar_rollout_mse:.6f}, "
                      f"late_mse={ar_late_mse:.6f} ({ar_time:.1f}s)")

                if wandb_run:
                    wandb_run.log({
                        'val/ar_rollout_mse': ar_rollout_mse,
                        'val/ar_late_mse': ar_late_mse,
                        'val/ar_eval_time': ar_time,
                    })

                # Best checkpoint by AR rollout
                if ar_rollout_mse < best_ar_mse:
                    best_ar_mse = ar_rollout_mse
                    _save_dyn(
                        'best_ar.pt',
                        val_loss=avg_val_loss,
                        ar_rollout_mse=ar_rollout_mse,
                        best_ar_mse=best_ar_mse)
                    print(f"  New best AR mse={ar_rollout_mse:.6f}")

            # Periodic checkpoint — written after this epoch's validation and
            # AR eval so a resume restores best_val_loss / best_ar_mse as of
            # the end of this epoch (not one eval behind).
            if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
                _save_dyn(
                    f'epoch_{epoch:04d}.pt',
                    optimizer=optimizer, scheduler=scheduler,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss, best_val_loss=best_val_loss,
                    best_ar_mse=best_ar_mse)

    if wandb_run:
        wandb_run.finish()

    teardown_ddp()

    if is_main:
        print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
        print(f"Checkpoints: {checkpoint_dir}")


if __name__ == '__main__':
    main()
