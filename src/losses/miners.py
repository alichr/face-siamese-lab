"""Triplet miners (plan §6b, poster panel 4b + the "hard negatives" tip).

Every image in the PK batch is an anchor; a miner decides which positive and
which negative it is paired with. That choice matters more than the margin does,
which is the point E3 measures.

Why mining exists at all: once training is underway, most randomly drawn
triplets already satisfy `d_ap + alpha < d_an`, so they produce exactly zero
loss and zero gradient. The batch still costs a full forward and backward pass.
Random mining therefore spends most of its compute on triplets that teach
nothing, and the `active fraction` curve collapses toward zero while the model
stops improving. Semi-hard and batch-hard keep the fraction high by selecting
triplets that still violate the margin.

The three miners (all return anchor/positive/negative index triples):

  random      uniform positive, uniform negative. The control condition.
  semi-hard   FaceNet: negatives with `d_ap < d_an < d_ap + alpha` -- further
              than the positive but still inside the margin. These give a
              nonzero, *bounded* gradient. Falls back to the hardest violating
              negative when no semi-hard one exists.
  batch-hard  Hermans et al.: hardest positive (max d_ap) and hardest negative
              (min d_an) per anchor. Strongest signal, but sensitive to label
              noise -- a mislabelled image is exactly what "hardest" selects.

Why not simply always take the hardest negative? Because the hardest negatives
early in training are often the model's worst mistakes, and pulling hard on them
collapses the embedding to a single point. Semi-hard exists to sit between
"useless" and "destructive": strictly further than the positive (so the model is
not asked to invert an ordering it already has right) but inside the margin (so
there is still something to fix).
"""

from __future__ import annotations

import torch

from src.losses.geometry import euclidean_distance_matrix

MINERS = ("random", "semi-hard", "batch-hard")


def _masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(positive_mask, negative_mask)`, both with the diagonal excluded."""
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    return same & ~eye, ~same


def _random_choice_from_mask(mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Pick one uniformly random True column per row.

    Implemented by scoring valid entries with uniform noise and taking the argmax,
    which is a single fused kernel rather than a Python loop over N anchors.
    Rows with no valid entry fall back to index 0; callers filter those out.
    """
    noise = torch.rand(mask.shape, generator=generator, device="cpu").to(mask.device)
    return (noise * mask).argmax(dim=1)


def mine(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    miner: str = "semi-hard",
    margin: float = 0.2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one (anchor, positive, negative) triplet per valid anchor.

    Args:
        embeddings: (N, d) L2-normalized embeddings.
        labels: (N,) identity labels.
        miner: one of `MINERS`.
        margin: alpha, used only by `semi-hard` to define the upper band.
        generator: RNG for the stochastic miners.

    Returns:
        `(anchor_idx, positive_idx, negative_idx)`, each (T,) with T <= N.
        Anchors lacking any positive or any negative in the batch are dropped.

    Raises:
        ValueError: on an unknown miner name.
    """
    if miner not in MINERS:
        raise ValueError(f"miner must be one of {MINERS}, got {miner!r}")

    if generator is None:
        generator = torch.Generator().manual_seed(0)

    distances = euclidean_distance_matrix(embeddings)
    positive_mask, negative_mask = _masks(labels)

    # An anchor needs at least one positive and one negative to form a triplet.
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        empty = torch.empty(0, dtype=torch.long, device=embeddings.device)
        return empty, empty, empty

    if miner == "batch-hard":
        # Hardest positive: the same-identity image the model places FURTHEST away.
        # Non-positives are pushed to -inf so they can never win the max.
        positive_idx = distances.masked_fill(~positive_mask, float("-inf")).argmax(dim=1)
        # Hardest negative: the different-identity image placed CLOSEST.
        negative_idx = distances.masked_fill(~negative_mask, float("inf")).argmin(dim=1)

    elif miner == "random":
        positive_idx = _random_choice_from_mask(positive_mask, generator)
        negative_idx = _random_choice_from_mask(negative_mask, generator)

    else:  # semi-hard
        positive_idx = _random_choice_from_mask(positive_mask, generator)
        d_ap = distances.gather(1, positive_idx[:, None])  # (N, 1)

        # The semi-hard band: strictly further than the positive, still inside
        # the margin. Both bounds are strict, per the plan's definition.
        semi_hard_mask = negative_mask & (distances > d_ap) & (distances < d_ap + margin)

        # Fallback for anchors with an empty band: the hardest *violating*
        # negative, i.e. the closest one that still produces nonzero loss
        # (d_an < d_ap + margin). If nothing violates, this picks the closest
        # negative, which correctly yields zero loss.
        violating = negative_mask & (distances < d_ap + margin)
        fallback_source = torch.where(violating.any(dim=1, keepdim=True), violating, negative_mask)
        fallback_idx = distances.masked_fill(~fallback_source, float("inf")).argmin(dim=1)

        semi_hard_idx = _random_choice_from_mask(semi_hard_mask, generator)
        has_semi_hard = semi_hard_mask.any(dim=1)
        negative_idx = torch.where(has_semi_hard, semi_hard_idx, fallback_idx)

    anchor_idx = torch.arange(len(labels), device=embeddings.device)
    return anchor_idx[valid], positive_idx[valid], negative_idx[valid]
