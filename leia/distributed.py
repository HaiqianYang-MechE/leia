"""DDP plumbing — initialization, rank queries, all-reduce helpers.

No new logic: collected from the inline patterns in `scripts/train_tokenizer.py`
and `scripts/train_dynamics.py` so future training scripts don't re-implement
each one.

Usage::

    from leia.distributed import setup_ddp, teardown_ddp, is_main_rank, all_reduce_avg

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    if is_main_rank():
        print('main rank only')
    teardown_ddp()
"""

import os

import torch
import torch.distributed as dist


def setup_ddp():
    """Initialize NCCL process group if running under torchrun, otherwise single-GPU.

    Returns
    -------
    (rank, world_size, local_rank) : (int, int, int)
        rank=0, world_size=1, local_rank=0 in single-GPU mode.
    """
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def teardown_ddp():
    """Destroy the process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank() -> bool:
    """True on rank 0, including in single-GPU mode."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def world_size() -> int:
    """Number of ranks. Returns 1 in single-GPU mode."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_reduce_avg(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce SUM then divide by world size — gives the cross-rank mean.

    No-op in single-GPU mode. The argument is modified in-place when
    distributed; the same tensor is returned for chaining.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def barrier():
    """Synchronize all ranks; no-op in single-GPU mode."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
