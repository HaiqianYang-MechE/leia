#!/usr/bin/env python
"""
Train the PerceiverTokenizer.

Reads a per-plate mmap dataset (`{plate_id}_{coords,outputs,fps_order,grad_u}.npy`)
and optimizes a recon + Sobolev (`grad-head` / `tet` / `autograd`) + Cauchy
stress loss. Always launch via `scripts/launch.sh` (handles conda, torchrun, DDP).

Usage:
    bash scripts/launch.sh train-tokenizer --gpus 8 --run-name v2-tok -- \
        --mmap-dir data/microplate_lattice/plate_dataset_curated_mmap_100k \
        --num-latents 512 --hidden-dim 1024 --n-input-frames 3 \
        --num-encoder-layers 6 --num-decoder-layers 0 \
        --stress-head --cauchy-stress-weight 1.0 \
        --batch-size 8 --accum-steps 2 --lr 1e-4 --max-steps 100000 \
        --bf16 --wandb
"""

import os
import sys
import time
import math
import argparse
import datetime
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.tokenizer import PerceiverTokenizer
from leia.config import TOKENIZER_CHECKPOINT_ROOT, WANDB_PROJECT_TOKENIZER
from leia.regime import MICROPLATE_LATTICE
from leia.datasets import MmapMultiStructureDataset, padding_collate
from leia.losses import (
    compute_whole_plate_tet_sobolev_loss,
    compute_autograd_stress_loss,
)
from leia.distributed import setup_ddp, teardown_ddp, is_main_rank
from leia.training import save_checkpoint, load_checkpoint, resume_training_state

# Architectural defaults used as argparse defaults (see configs/microplate_lattice/
# tokenizer_v2.yaml for production values; these are just CLI fallbacks).
TOKENIZER_NUM_LATENTS = 512
TOKENIZER_HIDDEN_DIM = 256
TOKENIZER_NUM_HEADS = 4
TOKENIZER_NUM_ENCODER_LAYERS = 6
TOKENIZER_NUM_DECODER_LAYERS = 2
TOKENIZER_DROPOUT = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Train PerceiverTokenizer")
    parser.add_argument('--run-name', type=str, required=True,
                        help='Run name; the checkpoint dir is <run-name>_<timestamp> (e.g. "v2-tok")')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='W&B project name (default: config WANDB_PROJECT_TOKENIZER)')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Stop after N optimizer steps (0=use --epochs)')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                        help='Save checkpoint every N optimizer steps')
    parser.add_argument('--val-interval', type=int, default=100,
                        help='Run validation every N optimizer steps (0=end of epoch only)')
    parser.add_argument('--num-latents', type=int, default=TOKENIZER_NUM_LATENTS)
    parser.add_argument('--hidden-dim', type=int, default=TOKENIZER_HIDDEN_DIM)
    parser.add_argument('--num-heads', type=int, default=TOKENIZER_NUM_HEADS,
                        help='Number of attention heads')
    parser.add_argument('--num-encoder-layers', type=int, default=TOKENIZER_NUM_ENCODER_LAYERS,
                        help='Number of encoder self-attention layers')
    parser.add_argument('--num-decoder-layers', type=int, default=TOKENIZER_NUM_DECODER_LAYERS,
                        help='Number of decoder self-attention layers')
    parser.add_argument('--dropout', type=float, default=TOKENIZER_DROPOUT,
                        help='Dropout rate for attention and FFN layers')
    parser.add_argument('--warmup-epochs', type=float, default=0,
                        help='Number of epochs for linear LR warmup (0=auto 5%% of total steps)')
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='LR warmup in optimizer steps (0=use --warmup-epochs or auto)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Load model weights only, reset optimizer/scheduler/step counters '
                             '(for fine-tuning with new loss or LR schedule)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--max-rves', type=int, default=0,
                        help='Max RVEs to load (0=all, for fast iteration)')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Gradient accumulation steps (effective_batch = batch_size * world_size * accum_steps)')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='Enable bf16 mixed precision (activates Flash Attention on H100)')
    parser.add_argument('--no-bf16', action='store_false', dest='bf16',
                        help='Disable bf16 (use fp32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker count (default 4)')
    parser.add_argument('--mmap-dir', type=str, default=None,
                        help='Memory-mapped numpy dir (from scripts/build_tokenizer_mmap.py). '
                             'All DDP processes share pages via OS page cache.')
    parser.add_argument('--sobolev-beta', type=float, default=0.0,
                        help='Weight for Sobolev gradient loss (0=disabled)')
    parser.add_argument('--sobolev-method', type=str, default='grad-head',
                        choices=['grad-head', 'tet', 'autograd'],
                        help='How the Sobolev loss computes the predicted '
                             'displacement gradient. grad-head (default): use '
                             'the model.grad_proj output (requires a precomputed '
                             'grad_u target). tet: '
                             'whole-plate FEM tet shape function diff of '
                             'decoded u over a Monte Carlo tet sample (requires '
                             '--tet-geom-dir). autograd: torch.autograd.grad '
                             'Jacobian of decoded u w.r.t. ref (requires '
                             'precomputed grad_u target).')
    parser.add_argument('--tet-geom-dir', type=str, default=None,
                        help='Per-plate tet geometry dir (from '
                             'scripts/precompute_tet_geom.py). Required '
                             'when --sobolev-method tet.')
    parser.add_argument('--tet-sample-k-core', type=int, default=300,
                        help='Deterministic FPS-tet core size per sample '
                             '(--sobolev-method tet only).')
    parser.add_argument('--tet-sample-k-rand', type=int, default=200,
                        help='Random remainder tet sample size per sample '
                             '(--sobolev-method tet only).')
    parser.add_argument('--sobolev-warmup-steps', type=int, default=0,
                        help='Linearly ramp Sobolev beta from 0 over N steps')
    parser.add_argument('--lr-schedule', choices=['cosine', 'constant'], default='cosine',
                        help='LR schedule after warmup: cosine (default) or constant')
    parser.add_argument('--recon-weight', type=float, default=1.0,
                        help='Weight for reconstruction loss (default 1.0, set 0 for physics-only)')
    # FPS subsampling flags
    parser.add_argument('--subsample-n', type=int, default=0,
                        help='Total subsampled nodes per mesh (0=off; paper: 10240 with --fps-core-n 8192)')
    parser.add_argument('--fps-core-n', type=int, default=0,
                        help='FPS core set size (0=all FPS, no random augmentation)')
    parser.add_argument('--stress-mu', type=float, default=MICROPLATE_LATTICE.mu,
                        help='Neo-Hookean mu for autograd stress loss '
                             '(default from microplate_lattice manifest)')
    parser.add_argument('--stress-lmbda', type=float, default=MICROPLATE_LATTICE.lmbda,
                        help='Neo-Hookean lambda for autograd stress loss '
                             '(default from microplate_lattice manifest)')
    parser.add_argument('--exclude-plates', type=str, default='',
                        help='Comma-separated plate / RVE IDs to exclude from the '
                             'mmap-discovered dataset (e.g. "mg_225,mg_385,mg_408,mg_416" '
                             'to drop the duplicate solid-plate group)')
    # Cauchy stress head flags
    parser.add_argument('--n-input-frames', type=int, default=1, choices=[1, 3],
                        help='Number of consecutive displacement frames to encode '
                             '(1=single frame, 3=temporal context for velocity/acceleration)')
    parser.add_argument('--stress-head', action='store_true',
                        help='Enable Cauchy stress prediction head (6-dim sym6)')
    parser.add_argument('--cauchy-stress-weight', type=float, default=0.0,
                        help='Weight for Cauchy stress head MSE loss (0=disabled)')
    parser.add_argument('--cauchy-stress-warmup-steps', type=int, default=0,
                        help='Linearly ramp Cauchy stress weight from 0 over N steps')
    # MicroPlate visco (load-reverse) data flags
    parser.add_argument('--visco-load-reverse-data', type=str, default=None,
                        help='Path to MicroPlate visco load-reverse HDF5')
    parser.add_argument('--splits-json', type=str, default=None,
                        help='Path to train/val/test splits JSON (visco only)')
    return parser.parse_args()






def main():
    args = parse_args()
    if not args.mmap_dir and not args.visco_load_reverse_data:
        raise ValueError(
            "Must provide either --mmap-dir <path> (MicroPlate lattice) or "
            "--visco-load-reverse-data <path.h5> --splits-json <splits.json> (MicroPlate visco)")

    rank, world_size, local_rank = setup_ddp()
    is_main = is_main_rank()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main:
        print(f"Device: {device}, World size: {world_size}")
        print(f"Data: {args.mmap_dir or args.visco_load_reverse_data}")

    # If resuming, extract disp_scale from checkpoint to ensure consistency.
    resume_disp_scale = None
    if args.resume:
        _ckpt_meta = torch.load(args.resume, map_location='cpu', weights_only=False)
        resume_disp_scale = _ckpt_meta.get('disp_scale', None)
        if resume_disp_scale is not None and is_main:
            print(f"Using checkpoint disp_scale={resume_disp_scale:.4f}")
        del _ckpt_meta

    # Dataset construction. Two paths:
    #   - args.visco_load_reverse_data + args.splits_json → MicroPlate visco
    #   - args.mmap_dir → MicroPlate lattice (production whole-plate mmap)
    if args.visco_load_reverse_data:
        from leia.datasets import make_visco_load_reverse_datasets
        if not args.splits_json:
            raise ValueError("--splits-json required with --visco-load-reverse-data")
        train_dataset, val_dataset = make_visco_load_reverse_datasets(
            args.visco_load_reverse_data, args.splits_json,
            disp_scale=resume_disp_scale,
            n_input_frames=args.n_input_frames)
        needs_padding = False
        if is_main:
            print(f"MicroPlate visco: {len(train_dataset)} train, {len(val_dataset)} val, "
                  f"disp_scale={train_dataset.disp_scale:.4f}, "
                  f"stress_mean={train_dataset.stress_mean.tolist()}, "
                  f"stress_std={train_dataset.stress_std.tolist()}")

    elif args.mmap_dir:
        # Memory-mapped whole-plate dataset: ranks share OS pages via page cache.
        sub_n = args.subsample_n
        fps_n = args.fps_core_n
        exclude_plates = (
            [p.strip() for p in args.exclude_plates.split(',') if p.strip()]
            if args.exclude_plates else None)
        full_ds = MmapMultiStructureDataset(
            args.mmap_dir, split="train", max_rves=args.max_rves,
            val_ratio=MICROPLATE_LATTICE.splits.random_val_ratio,
            seed=MICROPLATE_LATTICE.splits.random_split_seed,
            subsample_n=sub_n, fps_core_n=fps_n,
            exclude_plates=exclude_plates,
            n_input_frames=args.n_input_frames)
        disp_scale_val = full_ds.disp_scale
        mesh_coords = full_ds.mesh_coords
        mesh_ids = full_ds.mesh_ids
        mmap_outputs = full_ds.mmap_outputs
        mmap_grads = full_ds.mmap_grads
        fps_orders = full_ds.fps_orders
        coord_scales = full_ds.coord_scales
        full_ds_val = MmapMultiStructureDataset(
            args.mmap_dir, split="val", max_rves=args.max_rves,
            val_ratio=MICROPLATE_LATTICE.splits.random_val_ratio,
            seed=MICROPLATE_LATTICE.splits.random_split_seed,
            disp_scale=disp_scale_val,
            subsample_n=sub_n, fps_core_n=fps_n,
            exclude_plates=exclude_plates,
            n_input_frames=args.n_input_frames)

        # Cauchy sym6 normalization stats — rank 0 computes once, broadcasts.
        stress_mean = None
        stress_std = None
        if args.stress_head and args.cauchy_stress_weight > 0:
            stress_mean = torch.zeros(6, dtype=torch.float32)
            stress_std = torch.ones(6, dtype=torch.float32)
            if is_main:
                print(f"Precomputing Cauchy sym6 stats "
                      f"(mu={args.stress_mu}, lmbda={args.stress_lmbda}) "
                      f"on rank 0 (FPS-core subset, ~5K frames)...",
                      flush=True)
                t0_stats = time.time()
                m, s = full_ds.precompute_stress_stats(
                    n_samples=5000, mu=args.stress_mu, lmbda=args.stress_lmbda,
                    n_fps_core=args.fps_core_n or 8192)
                stress_mean = m
                stress_std = s
                print(f"  stress_mean = {stress_mean.tolist()}")
                print(f"  stress_std  = {stress_std.tolist()}")
                print(f"  stats precompute wall-clock: {time.time() - t0_stats:.1f}s",
                      flush=True)
            if world_size > 1:
                stress_mean = stress_mean.to(device)
                stress_std = stress_std.to(device)
                dist.broadcast(stress_mean, src=0)
                dist.broadcast(stress_std, src=0)
                stress_mean = stress_mean.cpu()
                stress_std = stress_std.cpu()
            full_ds.stress_mu = float(args.stress_mu)
            full_ds.stress_lmbda = float(args.stress_lmbda)

        train_dataset = MmapMultiStructureDataset.from_shared(
            full_ds.frames, mesh_coords, mesh_ids, mmap_outputs, disp_scale_val,
            mmap_grads=mmap_grads, fps_orders=fps_orders,
            coord_scales=coord_scales,
            subsample_n=sub_n, fps_core_n=fps_n,
            n_input_frames=args.n_input_frames)
        val_dataset = MmapMultiStructureDataset.from_shared(
            full_ds_val.frames, mesh_coords, mesh_ids, mmap_outputs, disp_scale_val,
            mmap_grads=mmap_grads, fps_orders=fps_orders,
            coord_scales=coord_scales,
            subsample_n=sub_n, fps_core_n=fps_n,
            n_input_frames=args.n_input_frames)
        if stress_mean is not None:
            train_dataset.set_stress_stats(
                stress_mean, stress_std,
                mu=args.stress_mu, lmbda=args.stress_lmbda)
            val_dataset.set_stress_stats(
                stress_mean, stress_std,
                mu=args.stress_mu, lmbda=args.stress_lmbda)
        # Whole-plate tet geometry — only when --sobolev-method tet
        if args.sobolev_method == 'tet':
            if not args.tet_geom_dir:
                raise ValueError(
                    "--sobolev-method tet requires --tet-geom-dir")
            train_dataset.attach_tet_geom(
                args.tet_geom_dir,
                k_core=args.tet_sample_k_core,
                k_rand=args.tet_sample_k_rand)
            val_dataset.attach_tet_geom(
                args.tet_geom_dir,
                k_core=args.tet_sample_k_core,
                k_rand=args.tet_sample_k_rand)
        del full_ds, full_ds_val
        needs_padding = True


    disp_scale = getattr(train_dataset, 'disp_scale', 1.0)
    if is_main:
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        print(f"Displacement scale: {disp_scale:.4f}")

    collate_fn = padding_collate if needs_padding else None
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = PerceiverTokenizer(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        num_latents=args.num_latents,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        stress_head=args.stress_head,
        n_input_frames=args.n_input_frames,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model params: {n_params/1e6:.2f}M")
        eff_batch = args.batch_size * world_size * args.accum_steps
        print(f"Effective batch size: {args.batch_size} x {world_size} GPUs x {args.accum_steps} accum = {eff_batch}")

    if world_size > 1:
        # find_unused_parameters: grad_proj head is created unconditionally on
        # the model but only enters the loss when sobolev-method is grad-head;
        # with stress-head + sobolev_beta=0 it is unused.
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Step-level warmup + cosine decay (counted in optimizer steps, not batches)
    steps_per_epoch = len(train_loader) // args.accum_steps
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    elif args.warmup_epochs > 0:
        warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    else:
        warmup_steps = max(100, total_steps // 20)  # 5% of total, floor 100
    if is_main:
        print(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if args.lr_schedule == 'constant':
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
        resume_global_step = rs['global_step']
        resume_best_val_loss = rs['best_val_loss']
        resume_best_val_recon = rs['best_val_recon']
    else:
        start_epoch = 0
        resume_global_step = 0
        resume_best_val_loss = float('inf')
        resume_best_val_recon = float('inf')

    # Checkpoint dir
    checkpoint_dir = os.path.join(
        TOKENIZER_CHECKPOINT_ROOT,
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
            project=args.wandb_project or WANDB_PROJECT_TOKENIZER,
            name=args.run_name,
            config=vars(args),
        )
        wandb_run.config.update({'disp_scale': disp_scale}, allow_val_change=True)

    # --- Sobolev gradient loss setup ---
    sobolev_beta = args.sobolev_beta
    sobolev_method = args.sobolev_method
    # Check if gradient head is available (model has grad_proj)
    raw_model_check = model.module if hasattr(model, 'module') else model
    has_grad_head = hasattr(raw_model_check, 'grad_proj')
    # Check if GT grad_u is available in dataset (for gradient head path)
    has_grad_u_gt = (hasattr(train_dataset, 'mmap_grads')
                     and train_dataset.mmap_grads)
    has_tet_geom = (hasattr(train_dataset, 'tet_geom')
                    and bool(train_dataset.tet_geom))
    if sobolev_beta > 0:
        if sobolev_method == 'grad-head':
            if not (has_grad_head and has_grad_u_gt):
                raise ValueError(
                    "--sobolev-method grad-head requires the tokenizer to have "
                    "a gradient head (--stress-head off, default) and the dataset "
                    "to expose grad_u ground truth via the mmap layout.")
            if is_main:
                print(f"Sobolev loss: method=grad-head, beta={sobolev_beta} "
                      f"({sum(p.numel() for p in raw_model_check.grad_proj.parameters())} params)")
        elif sobolev_method == 'tet':
            if not has_tet_geom:
                raise ValueError(
                    "--sobolev-method tet requires --tet-geom-dir with "
                    "precomputed plate tet geometry (and grad_u mmap loaded).")
            if not has_grad_u_gt:
                raise ValueError(
                    "--sobolev-method tet requires grad_u mmap; "
                    "set --mmap-dir to a dir containing {plate}_grad_u.npy files.")
            if is_main:
                print(f"Sobolev loss: method=tet, beta={sobolev_beta} "
                      f"(K_core={args.tet_sample_k_core}, "
                      f"K_rand={args.tet_sample_k_rand}, "
                      f"{len(train_dataset.tet_geom)} plates)")
        elif sobolev_method == 'autograd':
            if not has_grad_u_gt:
                raise ValueError(
                    "--sobolev-method autograd requires grad_u mmap.")
            if is_main:
                print(f"Sobolev loss: method=autograd, beta={sobolev_beta} "
                      f"(SDPA=MATH, create_graph=True).")

    # --- Validation helper (called mid-epoch and end-of-epoch) ---
    best_val_loss = resume_best_val_loss
    # Tracks lowest val_recon for the parallel `best_recon.pt` save.
    # Used for stress-accuracy regimes where cauchy_stress loss magnitude
    # dominates the combined selector (e.g. visco-LR) and `best.pt` gets
    # picked too early on a stress-loss local minimum. Restored on resume so
    # the first validation afterwards cannot overwrite best_recon.pt.
    best_val_recon = resume_best_val_recon

    def run_validation(epoch, global_step, train_recon_sum,
                       train_sobolev_sum, train_cauchy_stress_sum,
                       n_train_batches, t_start):
        nonlocal best_val_loss, best_val_recon
        model.eval()
        val_recon = 0.0
        val_sobolev = 0.0
        val_cauchy_stress = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                # padding_collate always returns a dict for both lattice mmap
                # (with P_gt / coord_scale) and visco (without).
                x = batch['x'].to(device)
                ref = batch['ref'].to(device)
                mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
                val_P_gt_batch = batch['P_gt'].to(device) if batch.get('P_gt') is not None else None
                val_coord_scale_batch = batch['coord_scale'].to(device) if batch.get('coord_scale') is not None else None

                # Reconstruction target (current frame only; multi-frame x is [B,N,3*nf])
                x_target = batch['x_target'].to(device) if batch.get('x_target') is not None else x

                val_stress_gt_batch = batch['stress_gt'].to(device) if batch.get('stress_gt') is not None else None

                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                    x_recon, grad_pred, stress_pred = model(x, ref, padding_mask=mask)
                    raw_loss = (x_recon - x_target) ** 2
                    raw_loss = raw_loss.sum(dim=-1)
                    if mask is not None:
                        valid = ~mask
                        recon_loss = (raw_loss * valid.float()).sum() / valid.sum()
                    else:
                        recon_loss = raw_loss.mean()

                    if sobolev_beta > 0 and sobolev_method == 'grad-head':
                        if grad_pred is not None and val_P_gt_batch is not None:
                            if val_coord_scale_batch is not None:
                                cs = val_coord_scale_batch[:, None, None, :]
                                grad_u_norm = val_P_gt_batch * cs / disp_scale
                            else:
                                grad_u_norm = val_P_gt_batch / disp_scale
                            if mask is not None:
                                valid = ~mask
                                diff = (grad_pred - grad_u_norm) ** 2
                                diff = diff.sum(dim=(-2, -1))
                                s_loss = (diff * valid.float()).sum() / valid.sum()
                            else:
                                s_loss = F.mse_loss(grad_pred, grad_u_norm)
                            val_sobolev += s_loss.item()
                    elif (sobolev_beta > 0 and sobolev_method == 'tet'
                          and batch.get('tet_sample') is not None
                          and val_P_gt_batch is not None
                          and val_coord_scale_batch is not None):
                        s_loss = compute_whole_plate_tet_sobolev_loss(
                            x_recon, val_P_gt_batch, batch['tet_sample'],
                            val_coord_scale_batch, disp_scale)
                        val_sobolev += s_loss.item()
                    elif (sobolev_beta > 0 and sobolev_method == 'autograd'
                          and val_P_gt_batch is not None
                          and val_coord_scale_batch is not None):
                        # autograd Sobolev needs grad-enabled context internally
                        # (it calls requires_grad_(True) on ref); val loop is in
                        # no_grad, so re-enable here.
                        with torch.enable_grad():
                            with torch.autocast('cuda', enabled=False):
                                s_loss = compute_autograd_stress_loss(
                                    model, x, ref, val_P_gt_batch,
                                    val_coord_scale_batch, disp_scale)
                        val_sobolev += s_loss.item()

                    if args.cauchy_stress_weight > 0 and stress_pred is not None and val_stress_gt_batch is not None:
                        cs_loss = F.mse_loss(stress_pred, val_stress_gt_batch)
                        val_cauchy_stress += cs_loss.item()

                val_recon += recon_loss.item()
                val_n += 1
        model.train()

        avg_train_recon = train_recon_sum / max(n_train_batches, 1)
        avg_train_sobolev = train_sobolev_sum / max(n_train_batches, 1)
        avg_train_cauchy_stress = train_cauchy_stress_sum / max(n_train_batches, 1)
        # Aggregate validation metrics across DDP ranks
        if world_size > 1:
            val_tensors = torch.tensor(
                [val_recon, val_sobolev, val_cauchy_stress, val_n],
                dtype=torch.float64, device=device)
            dist.all_reduce(val_tensors)
            (val_recon, val_sobolev, val_cauchy_stress, val_n) = val_tensors.tolist()

        avg_val_recon = val_recon / max(val_n, 1)
        avg_val_sobolev = val_sobolev / max(val_n, 1)
        avg_val_cauchy_stress = val_cauchy_stress / max(val_n, 1)
        # best.pt selector mirrors the training objective at *target* weights
        # (not warmup-ramped) so early-step checkpoints whose stress head hasn't
        # warmed up don't unfairly win on stress-light loss.
        avg_val_loss = (
            args.recon_weight * avg_val_recon
            + sobolev_beta * avg_val_sobolev
            + args.cauchy_stress_weight * avg_val_cauchy_stress
        )
        dt = time.time() - t_start

        if is_main:
            lr = optimizer.param_groups[0]['lr']
            sob_str = f" sob={avg_val_sobolev:.6f}" if sobolev_beta > 0 else ""
            cauchy_str = f" cauchy={avg_val_cauchy_stress:.6f}" if args.cauchy_stress_weight > 0 else ""
            print(f"Step {global_step:6d} (ep {epoch}) | "
                  f"train recon={avg_train_recon:.6f} | "
                  f"val recon={avg_val_recon:.6f}"
                  f"{sob_str}{cauchy_str} | "
                  f"lr={lr:.2e} | {dt:.1f}s")

            if wandb_run:
                log_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'train/recon_loss': avg_train_recon,
                    'val/recon_loss': avg_val_recon,
                    'val/total_loss': avg_val_loss,
                    'lr': lr,
                }
                if sobolev_beta > 0:
                    log_dict['train/sobolev'] = avg_train_sobolev
                    log_dict['val/sobolev'] = avg_val_sobolev
                if args.cauchy_stress_weight > 0:
                    log_dict['train/cauchy_stress'] = avg_train_cauchy_stress
                    log_dict['val/cauchy_stress'] = avg_val_cauchy_stress
                wandb_run.log(log_dict)

            # Best model — combined-loss selector (recon + stress + sobolev).
            # Right when loss magnitudes are commensurate (e.g. lattice). For
            # stress-accuracy regimes where stress >> recon (e.g. visco-LR),
            # the parallel best_recon.pt below is the paper-faithful selector.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                extras = {
                    'global_step': global_step,
                    'val_recon_loss': avg_val_recon,
                    'val_total_loss': avg_val_loss,
                    'best_val_loss': best_val_loss,
                    'disp_scale': disp_scale,
                }
                if hasattr(train_dataset, 'stress_mean') and train_dataset.stress_mean is not None:
                    extras['stress_mean'] = train_dataset.stress_mean.numpy()
                    extras['stress_std'] = train_dataset.stress_std.numpy()
                save_checkpoint(
                    os.path.join(checkpoint_dir, 'best.pt'),
                    model, epoch=epoch, config=vars(args), extras=extras)
                print(f"  New best val_loss={best_val_loss:.6f}")

            # Parallel best_recon.pt — selects on val_recon alone.
            if avg_val_recon < best_val_recon:
                best_val_recon = avg_val_recon
                extras = {
                    'global_step': global_step,
                    'val_recon_loss': avg_val_recon,
                    'val_total_loss': avg_val_loss,
                    'best_val_recon': best_val_recon,
                    'disp_scale': disp_scale,
                }
                if hasattr(train_dataset, 'stress_mean') and train_dataset.stress_mean is not None:
                    extras['stress_mean'] = train_dataset.stress_mean.numpy()
                    extras['stress_std'] = train_dataset.stress_std.numpy()
                save_checkpoint(
                    os.path.join(checkpoint_dir, 'best_recon.pt'),
                    model, epoch=epoch, config=vars(args), extras=extras)
                print(f"  New best val_recon={best_val_recon:.6f}")

        return avg_val_loss

    # Training loop
    global_step = resume_global_step
    last_val_step = -1

    def _step_extras(global_step):
        """Build the per-step `extras` dict (the best.pt save inside
        `run_validation` builds the same dict with its own val metrics)."""
        extras = {
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'best_val_recon': best_val_recon,
            'disp_scale': disp_scale,
        }
        if hasattr(train_dataset, 'stress_mean') and train_dataset.stress_mean is not None:
            extras['stress_mean'] = train_dataset.stress_mean.numpy()
            extras['stress_std'] = train_dataset.stress_std.numpy()
        return extras

    for epoch in itertools.count(start_epoch):
        # Termination checks
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
        if args.max_steps == 0 and epoch >= args.epochs:
            break

        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_recon = 0.0
        total_sobolev = 0.0
        total_cauchy_stress = 0.0
        n_batches = 0
        done = False

        accum_steps = args.accum_steps
        for batch_idx, batch in enumerate(train_loader):
            # padding_collate always returns a dict.
            x = batch['x'].to(device)
            ref = batch['ref'].to(device)
            mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
            P_gt_batch = batch['P_gt'].to(device) if batch.get('P_gt') is not None else None
            coord_scale_batch = batch['coord_scale'].to(device) if batch.get('coord_scale') is not None else None

            # Reconstruction target: always [B, N, 3] (current frame only).
            # For multi-frame input, x is [B, N, 3*nf] but x_target is [B, N, 3].
            x_target = batch['x_target'].to(device) if batch.get('x_target') is not None else x

            # Cauchy stress GT (visco only — lattice path uses on-the-fly grad_u → Cauchy).
            stress_gt_batch = batch['stress_gt'].to(device) if batch.get('stress_gt') is not None else None

            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                x_recon, grad_pred, stress_pred = model(x, ref, padding_mask=mask)

                # Masked MSE reconstruction loss
                raw_loss = (x_recon - x_target) ** 2
                raw_loss = raw_loss.sum(dim=-1)  # [B, N]

                if mask is not None:
                    valid = ~mask
                    recon_loss = (raw_loss * valid.float()).sum() / valid.sum()
                else:
                    recon_loss = raw_loss.mean()

                # Effective Sobolev beta with optional warmup
                sob_beta_eff = sobolev_beta
                if sobolev_beta > 0 and args.sobolev_warmup_steps > 0:
                    sob_beta_eff = min(1.0, global_step / args.sobolev_warmup_steps) * sobolev_beta

                # Sobolev loss — dispatch by --sobolev-method.
                #   grad-head: model.grad_proj output supervised with precomputed grad_u
                #              (~zero overhead).
                #   tet:       FEM tet shape function diff of decoded u over a Monte
                #              Carlo tet sample, supervised against precomputed grad_u
                #              averaged to per-tet (whole-plate).
                #   autograd:  torch.autograd Jacobian of decoded u w.r.t. ref, computed
                #              by a separate decode pass under SDPA=MATH.
                sobolev_loss = None
                if sob_beta_eff > 0 and sobolev_method == 'grad-head':
                    if grad_pred is not None and P_gt_batch is not None:
                        # Gradient head: direct prediction, zero overhead.
                        if coord_scale_batch is not None:
                            cs = coord_scale_batch[:, None, None, :]  # [B, 1, 1, 3]
                            grad_u_norm = P_gt_batch * cs / disp_scale
                        else:
                            grad_u_norm = P_gt_batch / disp_scale
                        if mask is not None:
                            valid = ~mask
                            diff = (grad_pred - grad_u_norm) ** 2
                            diff = diff.sum(dim=(-2, -1))  # [B, N]
                            sobolev_loss = (diff * valid.float()).sum() / valid.sum()
                        else:
                            sobolev_loss = F.mse_loss(grad_pred, grad_u_norm)
                elif (sob_beta_eff > 0 and sobolev_method == 'tet'
                      and batch.get('tet_sample') is not None
                      and P_gt_batch is not None
                      and coord_scale_batch is not None):
                    sobolev_loss = compute_whole_plate_tet_sobolev_loss(
                        x_recon, P_gt_batch, batch['tet_sample'],
                        coord_scale_batch, disp_scale)
                elif (sob_beta_eff > 0 and sobolev_method == 'autograd'
                      and P_gt_batch is not None
                      and coord_scale_batch is not None):
                    # Re-decode under SDPA=MATH with create_graph=True. Operates
                    # at the same FPS sub-sample as recon (full-mesh autograd
                    # OOMs). The function does its own no_grad encode + MATH
                    # decode internally.
                    with torch.autocast('cuda', enabled=False):
                        sobolev_loss = compute_autograd_stress_loss(
                            model, x, ref, P_gt_batch, coord_scale_batch,
                            disp_scale)

                # Cauchy stress head loss (direct supervision, normalized MSE).
                cauchy_stress_weight_eff = args.cauchy_stress_weight
                if args.cauchy_stress_weight > 0 and args.cauchy_stress_warmup_steps > 0:
                    cauchy_stress_weight_eff = min(
                        1.0, global_step / args.cauchy_stress_warmup_steps
                    ) * args.cauchy_stress_weight
                cauchy_stress_loss = None
                if cauchy_stress_weight_eff > 0 and stress_pred is not None and stress_gt_batch is not None:
                    cauchy_stress_loss = F.mse_loss(stress_pred, stress_gt_batch)

                loss = args.recon_weight * recon_loss
                if sobolev_loss is not None:
                    loss = loss + sob_beta_eff * sobolev_loss
                if cauchy_stress_loss is not None:
                    loss = loss + cauchy_stress_weight_eff * cauchy_stress_loss
                loss = loss / accum_steps

            loss.backward()

            total_recon += recon_loss.item()
            if sobolev_loss is not None:
                total_sobolev += sobolev_loss.item()
            if cauchy_stress_loss is not None:
                total_cauchy_stress += cauchy_stress_loss.item()
            n_batches += 1

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # W&B train step logging
                if wandb_run and global_step % args.log_interval == 0:
                    step_log = {
                        'train/loss_step': loss.item() * accum_steps,
                        'train/recon_step': recon_loss.item(),
                        'lr': optimizer.param_groups[0]['lr'],
                    }
                    if sobolev_loss is not None:
                        step_log['train/sobolev_step'] = sobolev_loss.item()
                        step_log['train/sobolev_beta_eff'] = sob_beta_eff
                    if cauchy_stress_loss is not None:
                        step_log['train/cauchy_stress_step'] = cauchy_stress_loss.item()
                        step_log['train/cauchy_stress_weight_eff'] = cauchy_stress_weight_eff
                    wandb_run.log(step_log)

                # Step-based validation
                if (args.val_interval > 0 and
                        global_step % args.val_interval == 0):
                    run_validation(epoch, global_step, total_recon,
                                   total_sobolev, total_cauchy_stress,
                                   n_batches, t0)
                    last_val_step = global_step
                    # Reset running averages so next interval is fresh
                    total_recon = 0.0
                    total_sobolev = 0.0
                    total_cauchy_stress = 0.0
                    n_batches = 0
                    t0 = time.time()

                # Step-based checkpoint
                if (args.checkpoint_interval > 0 and
                        global_step % args.checkpoint_interval == 0
                        and is_main):
                    ckpt_path = os.path.join(checkpoint_dir, f'step_{global_step:06d}.pt')
                    save_checkpoint(
                        ckpt_path, model, epoch=epoch, config=vars(args),
                        optimizer=optimizer, scheduler=scheduler,
                        extras=_step_extras(global_step))
                    print(f"  Saved checkpoint: {ckpt_path}")

                # Max steps reached
                if args.max_steps > 0 and global_step >= args.max_steps:
                    done = True
                    break

        # End-of-epoch validation (skip if we just validated at this step)
        if last_val_step != global_step:
            run_validation(epoch, global_step, total_recon,
                           total_sobolev, total_cauchy_stress,
                           n_batches, t0)
            last_val_step = global_step

        if done:
            break

    # Final checkpoint at end of training
    if (global_step > 0 and is_main
            and (args.checkpoint_interval == 0
                 or global_step % args.checkpoint_interval != 0)):
        ckpt_path = os.path.join(checkpoint_dir, f'step_{global_step:06d}.pt')
        save_checkpoint(
            ckpt_path, model, epoch=epoch, config=vars(args),
            optimizer=optimizer, scheduler=scheduler,
            extras=_step_extras(global_step))
        print(f"  Saved checkpoint: {ckpt_path}")

    if wandb_run:
        wandb_run.finish()

    teardown_ddp()

    if is_main:
        print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")
        print(f"Checkpoints: {checkpoint_dir}")


if __name__ == '__main__':
    main()
