"""Pairwise contrastive loss, Hadsell et al. 2006 (plan §6a, poster panel 4a).

    L(z1, z2, y) = y * d^2  +  (1 - y) * max(0, m - d)^2

with d = ||zhat1 - zhat2||_2 and default m = 1.0.

Read the two branches separately:

  * y = 1 (same identity): the loss is d^2, minimized only at d = 0. Positives
    are pulled together with no floor -- there is no "close enough".
  * y = 0 (different identity): the loss is (m - d)^2 but only while d < m. Once
    a negative is at least m away it contributes exactly zero and stops
    producing gradient. The margin is what stops the model from spending
    capacity pushing already-distant negatives further apart.

Because embeddings are normalized, d is bounded by 2. Any m > 2 is unsatisfiable
-- every negative pair would contribute loss forever -- which is why E2 sweeps
m over {0.25, 0.5, 1.0, 1.5} and not past 2.
"""

from __future__ import annotations

import torch
from torch import nn

from src.losses.geometry import euclidean_distance_matrix

# Max distance between two points on the unit sphere.
MAX_NORMALIZED_DISTANCE: float = 2.0


def contrastive_loss_from_distances(
    distances: torch.Tensor, labels: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Per-pair contrastive loss from precomputed distances.

    The exact formula above, kept separate from batch construction so the unit
    vectors in plan §12 can be checked against it directly.

    Args:
        distances: (P,) pairwise distances d.
        labels: (P,) with 1 for same-identity, 0 for different-identity.
        margin: m, the negative-pair margin.

    Returns:
        (P,) per-pair losses (not reduced).
    """
    y = labels.to(distances.dtype)
    positive_term = y * distances.pow(2)
    negative_term = (1.0 - y) * torch.clamp(margin - distances, min=0.0).pow(2)
    return positive_term + negative_term


class ContrastiveLoss(nn.Module):
    """Contrastive loss over a PK batch, with the diagnostics the poster implies.

    Pairs are built inside the batch (plan §4): all same-identity pairs give y=1,
    and an equal number of randomly drawn cross-identity pairs give y=0. The
    balancing is deliberate -- a PK batch of P=64, K=4 contains 384 positive
    pairs but 31,872 negative ones, and leaving that 83:1 imbalance in place
    would let the negative term dominate the gradient entirely.

    Logged per step (plan §6a):
        mean_d_pos    mean distance over positive pairs -- should fall
        mean_d_neg    mean distance over negative pairs -- should rise toward m
        frac_neg_in_margin  fraction of negatives still inside m, i.e. still
                      contributing gradient. When this hits 0 the negative term
                      is finished and only positives are still being optimized.
    """

    def __init__(self, margin: float = 1.0, seed: int = 0) -> None:
        """
        Args:
            margin: m, default 1.0.
            seed: RNG seed for negative-pair sampling.

        Raises:
            ValueError: if margin <= 0 or margin > 2 (unsatisfiable on the sphere).
        """
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0, got {margin}")
        if margin > MAX_NORMALIZED_DISTANCE:
            raise ValueError(
                f"margin={margin} exceeds the maximum distance {MAX_NORMALIZED_DISTANCE} "
                "between normalized embeddings -- no negative pair could ever satisfy it."
            )
        self.margin = margin
        self._generator = torch.Generator().manual_seed(seed)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss over a PK batch.

        Args:
            embeddings: (N, d) embeddings, assumed L2-normalized by the encoder.
            labels: (N,) identity labels.

        Returns:
            `(loss, logs)` -- scalar mean loss and the per-step diagnostics.
        """
        distance_matrix = euclidean_distance_matrix(embeddings)
        n = embeddings.shape[0]

        same_identity = labels[:, None] == labels[None, :]
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=embeddings.device), diagonal=1)

        positive_mask = same_identity & upper
        negative_mask = (~same_identity) & upper

        pos_i, pos_j = positive_mask.nonzero(as_tuple=True)
        neg_i, neg_j = negative_mask.nonzero(as_tuple=True)

        if pos_i.numel() == 0 or neg_i.numel() == 0:
            # Degenerate batch (K=1, or a batch of one identity). Return a
            # zero that still carries grad_fn so the training step is well-formed.
            zero = embeddings.sum() * 0.0
            return zero, {"mean_d_pos": 0.0, "mean_d_neg": 0.0, "frac_neg_in_margin": 0.0}

        # Balance: subsample negatives down to the positive count (plan §4).
        n_keep = min(pos_i.numel(), neg_i.numel())
        if neg_i.numel() > n_keep:
            perm = torch.randperm(neg_i.numel(), generator=self._generator)[:n_keep]
            perm = perm.to(neg_i.device)
            neg_i, neg_j = neg_i[perm], neg_j[perm]

        d_pos = distance_matrix[pos_i, pos_j]
        d_neg = distance_matrix[neg_i, neg_j]

        distances = torch.cat([d_pos, d_neg])
        pair_labels = torch.cat(
            [torch.ones_like(d_pos), torch.zeros_like(d_neg)]
        )

        loss = contrastive_loss_from_distances(distances, pair_labels, self.margin).mean()

        logs = {
            "mean_d_pos": d_pos.mean().item(),
            "mean_d_neg": d_neg.mean().item(),
            "frac_neg_in_margin": (d_neg < self.margin).float().mean().item(),
        }
        return loss, logs
