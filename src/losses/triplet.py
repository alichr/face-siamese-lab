"""Triplet loss, Schroff et al. 2015 / FaceNet (plan §6b, poster panel 4b).

    L(a, p, n) = max(0, d(a,p) - d(a,n) + alpha)

Default alpha = 0.2.

The contrast with contrastive loss is the whole reason both are in this lab.
Contrastive fixes *absolute* targets: positives to distance 0, negatives to at
least m. Triplet fixes only a *relative* ordering -- the anchor's positive must
be closer than its negative by alpha, and where either sits in absolute terms is
unconstrained. That is a weaker requirement, and weaker is often better here:
faces vary in how tightly one identity can cluster, and forcing every identity to
the same absolute radius wastes capacity fighting that.

Once `d_ap + alpha < d_an` the triplet is satisfied and contributes exactly zero
loss and zero gradient. The *fraction of active (non-zero) triplets* is therefore
the single most instructive curve in this project (plan §6b): it shows directly
how much of each batch is still teaching the model anything, and it is what
separates the three miners in E3.
"""

from __future__ import annotations

import torch
from torch import nn

from src.losses.geometry import euclidean_distance_matrix
from src.losses.miners import mine


def triplet_loss_from_distances(
    d_ap: torch.Tensor, d_an: torch.Tensor, margin: float = 0.2
) -> torch.Tensor:
    """Per-triplet hinge loss `max(0, d_ap - d_an + alpha)`.

    Kept separate from mining so the plan §12 vectors check this directly.

    Args:
        d_ap: (T,) anchor-positive distances.
        d_an: (T,) anchor-negative distances.
        margin: alpha.

    Returns:
        (T,) per-triplet losses (not reduced).
    """
    return torch.clamp(d_ap - d_an + margin, min=0.0)


class TripletLoss(nn.Module):
    """Triplet loss over a PK batch with a configurable miner.

    Logged per step (plan §6b):
        active_fraction  fraction of mined triplets with non-zero loss. Watch
                         this collapse under `random` mining while `semi-hard`
                         and `batch-hard` hold it up -- that gap is E3's result.
        mean_d_ap / mean_d_an   the two distances the hinge compares.
        mean_violation   mean of (d_ap - d_an + alpha) over active triplets only,
                         i.e. how badly the surviving triplets are violated.
    """

    def __init__(self, margin: float = 0.2, miner: str = "semi-hard", seed: int = 0) -> None:
        """
        Args:
            margin: alpha, default 0.2 (E3 sweeps 0.1 / 0.2 / 0.4).
            miner: `random`, `semi-hard`, or `batch-hard`.
            seed: RNG seed for the stochastic miners.

        Raises:
            ValueError: if margin <= 0.
        """
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0, got {margin}")
        self.margin = margin
        self.miner = miner
        self._generator = torch.Generator().manual_seed(seed)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Mine triplets from the batch and compute the mean hinge loss.

        Args:
            embeddings: (N, d) L2-normalized embeddings.
            labels: (N,) identity labels.

        Returns:
            `(loss, logs)`.
        """
        anchor_idx, positive_idx, negative_idx = mine(
            embeddings, labels, self.miner, self.margin, self._generator
        )

        if anchor_idx.numel() == 0:
            zero = embeddings.sum() * 0.0
            return zero, {
                "active_fraction": 0.0,
                "mean_d_ap": 0.0,
                "mean_d_an": 0.0,
                "mean_violation": 0.0,
            }

        distances = euclidean_distance_matrix(embeddings)
        d_ap = distances[anchor_idx, positive_idx]
        d_an = distances[anchor_idx, negative_idx]

        per_triplet = triplet_loss_from_distances(d_ap, d_an, self.margin)

        # Mean over ALL mined triplets, not only the active ones. Averaging over
        # actives alone would rescale the gradient as the active fraction falls,
        # masking exactly the effect E3 sets out to measure.
        loss = per_triplet.mean()

        active = per_triplet > 0
        logs = {
            "active_fraction": active.float().mean().item(),
            "mean_d_ap": d_ap.mean().item(),
            "mean_d_an": d_an.mean().item(),
            "mean_violation": (
                per_triplet[active].mean().item() if active.any() else 0.0
            ),
        }
        return loss, logs
