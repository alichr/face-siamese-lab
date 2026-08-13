"""Alignment and uniformity, Wang & Isola 2020 (plan §8.4).

These two numbers explain *why* InfoNCE behaves as it does, which no accuracy
figure can. Wang & Isola showed the InfoNCE objective decomposes asymptotically
into exactly these competing terms:

    alignment   = E_pos ||zhat_i - zhat_j||^2
                  how tightly positive pairs sit together. LOWER is better.

    uniformity  = log E_{i,j} exp(-2 ||zhat_i - zhat_j||^2)
                  how evenly the embeddings spread over the hypersphere.
                  LOWER (more negative) is better -- it is the log of a Gaussian
                  potential, minimized by a uniform distribution.

The pair is diagnostic because the failure modes are opposite and both look bad
on accuracy alone:

  * collapse -- everything maps to one point. Alignment is perfect (0) and
    uniformity is terrible (near 0, its maximum). This is the classic failure of
    a contrastive loss with too small a margin, and alignment alone would call
    it a success.
  * dispersion -- embeddings spread beautifully but positives are not together.
    Uniformity is excellent, alignment poor.

A good encoder needs both, and the V6 scatter plots each checkpoint as a point
in that plane. The poster's claim that InfoNCE "transfers better" predicts it
should sit toward the lower-left corner relative to the pairwise losses.
"""

from __future__ import annotations

import numpy as np
import torch

from src.losses.geometry import l2_normalize


def alignment(
    embeddings: torch.Tensor, labels: np.ndarray, max_pairs: int = 50_000, seed: int = 0
) -> float:
    """Mean squared distance between positive pairs. Lower is better.

    Args:
        embeddings: (N, d) embeddings.
        labels: (N,) identity labels.
        max_pairs: cap on sampled positive pairs (the full set is O(N·K²)).
        seed: RNG seed for subsampling.

    Returns:
        `E_pos ||zhat_i - zhat_j||^2`.

    Raises:
        ValueError: if no identity has 2+ images.
    """
    z = l2_normalize(embeddings)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    left: list[int] = []
    right: list[int] = []
    for identity in np.unique(labels):
        rows = np.flatnonzero(labels == identity)
        if len(rows) < 2:
            continue
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left.append(int(rows[i]))
                right.append(int(rows[j]))

    if not left:
        raise ValueError("no positive pairs: every identity has a single image")

    if len(left) > max_pairs:
        keep = rng.choice(len(left), size=max_pairs, replace=False)
        left = [left[i] for i in keep]
        right = [right[i] for i in keep]

    difference = z[left] - z[right]
    return float(difference.pow(2).sum(dim=1).mean().item())


def uniformity(embeddings: torch.Tensor, t: float = 2.0, max_samples: int = 4096, seed: int = 0) -> float:
    """Log of the mean Gaussian potential. Lower (more negative) is better.

    Args:
        embeddings: (N, d) embeddings.
        t: the exponent scale; 2.0 is Wang & Isola's default.
        max_samples: subsample cap (the computation is O(N²)).
        seed: RNG seed for subsampling.

    Returns:
        `log E_{i,j} exp(-t ||zhat_i - zhat_j||^2)` over distinct pairs.
    """
    z = l2_normalize(embeddings)
    if z.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(z.shape[0], size=max_samples, replace=False)
        z = z[keep]

    squared = torch.cdist(z, z).pow(2)
    n = z.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool)

    # logsumexp over distinct pairs, minus log(count), for numerical stability.
    values = (-t * squared)[off_diagonal]
    return float((torch.logsumexp(values, dim=0) - np.log(values.numel())).item())


def align_uniform(
    embeddings: torch.Tensor, labels: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """Both diagnostics in one call (plan §8.4)."""
    return {
        "alignment": alignment(embeddings, labels, seed=seed),
        "uniformity": uniformity(embeddings, seed=seed),
    }
