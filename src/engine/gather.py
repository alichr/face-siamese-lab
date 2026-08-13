"""Gradient-preserving cross-GPU gather for global InfoNCE negatives (plan §6c).

This module exists because its failure mode is **silent**. With a detached
gather, training still runs, the loss still falls, and every curve looks healthy
-- but gradients from other ranks' anchors never reach the local embeddings, so
the objective being optimized is small-batch InfoNCE while the logs claim
large-batch InfoNCE. Experiment E5 (accuracy vs number of negatives) would then
measure nothing at all.

Why the gradient matters, concretely. For anchor i and any NEGATIVE j,

    dL_i/dz_j = (1/tau) * p_ij * z_i,     p_ij = softmax_j(s_ij / tau)

which is nonzero for every j in the denominator. Negatives are not passive
context -- the loss actively pushes them away, and that push is a real gradient
that must reach whichever GPU owns z_j. `torch.distributed.all_gather` returns
tensors with no autograd history, so that push is silently discarded: the
local embeddings still receive gradient from *local* anchors, so nothing
crashes and nothing looks wrong.

`torch.distributed.nn.functional.all_gather` keeps the graph. Its backward
reduce-scatters the incoming gradients, so rank r receives the SUM over all
ranks of the gradient w.r.t. its own slice -- which is exactly what the
single-process computation would have produced, provided each rank scales its
local loss by 1/world_size before calling backward. See `tests/ddp_equivalence.py`
for the derivation made executable.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


def is_distributed() -> bool:
    """True when a process group is initialized and has more than one rank."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0


def get_world_size() -> int:
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather `tensor` along dim 0, **preserving autograd**.

    Args:
        tensor: (L, ...) local tensor, identical shape on every rank.

    Returns:
        (L * world_size, ...) concatenated in rank order, so rank r's rows occupy
        `[r*L : (r+1)*L]`. That layout is what makes `rank_offset = rank * L` the
        correct self-masking offset in the global similarity matrix.

    Note:
        Returns the input unchanged when not running distributed, so the same
        code path serves single-GPU runs.
    """
    if not is_distributed():
        return tensor

    gathered = dist_nn.all_gather(tensor)
    return torch.cat(gathered, dim=0)


def gather_no_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather WITHOUT autograd -- the classic bug, kept for the negative control.

    Used only by the Phase 4 test to demonstrate that the gradient check has
    teeth. Never call this from training code: it is the exact mistake the
    critical gate exists to catch.
    """
    if not is_distributed():
        return tensor

    world_size = dist.get_world_size()
    buffer = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(buffer, tensor)
    # The local slice is replaced by a detached copy too -- this is what makes
    # the whole gathered tensor gradient-free.
    return torch.cat(buffer, dim=0)


def gather_labels(labels: torch.Tensor) -> torch.Tensor:
    """All-gather integer labels. No gradient is involved, so plain all_gather is correct."""
    if not is_distributed():
        return labels

    world_size = dist.get_world_size()
    buffer = [torch.zeros_like(labels) for _ in range(world_size)]
    dist.all_gather(buffer, labels)
    return torch.cat(buffer, dim=0)


def global_infonce_inputs(
    embeddings: torch.Tensor, labels: torch.Tensor, detach: bool = False
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prepare the (pool, pool_labels, rank_offset) triple for global InfoNCE.

    The embeddings must already be L2-normalized by the encoder **before** this
    call. Normalizing after the gather would be wrong in a way that is easy to
    miss: each rank would then renormalize rows it does not own, and the
    resulting graph would attribute those operations to the wrong device.

    Args:
        embeddings: (L, d) local normalized embeddings.
        labels: (L,) local labels.
        detach: use the gradient-free gather (negative control only).

    Returns:
        `(pool, pool_labels, rank_offset)` where `rank_offset = rank * L` is the
        row index of this rank's first anchor inside the gathered pool.
    """
    gather = gather_no_grad if detach else gather_with_grad
    pool = gather(embeddings)
    pool_labels = gather_labels(labels)
    rank_offset = get_rank() * embeddings.shape[0]
    return pool, pool_labels, rank_offset
