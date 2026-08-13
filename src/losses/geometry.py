"""Similarity and distance primitives shared by every loss (plan §5).

The whole lab rests on one identity. For L2-normalized embeddings,

    d(z1, z2)^2 = ||z1 - z2||^2
                = ||z1||^2 + ||z2||^2 - 2 z1.z2
                = 1 + 1 - 2 s(z1, z2)
                = 2 - 2 s(z1, z2)

which is why the poster's "cosine similarity" box and "Euclidean distance" box
are two views of the same quantity: d is a strictly decreasing function of s on
the unit sphere, so thresholding one is thresholding the other. Contrastive and
triplet losses are written in terms of d; InfoNCE in terms of s; on normalized
embeddings they are ranking the same pairs.

`tests/test_geometry.py` asserts the identity numerically (atol 1e-5).
"""

from __future__ import annotations

import torch

# Guards 1/||z|| when an embedding collapses to ~0. Large enough to matter in
# fp16/bf16, small enough not to perturb the fp32 unit-norm assertion at 1e-5.
_EPS: float = 1e-12


def l2_normalize(z: torch.Tensor, dim: int = -1, eps: float = _EPS) -> torch.Tensor:
    """Project embeddings onto the unit hypersphere: zhat = z / ||z||_2 (poster panel 1).

    Args:
        z: embeddings, any shape; normalized along `dim`.
        dim: dimension to normalize over (default: last, the feature dim).
        eps: floor on the norm, preventing division by zero.

    Returns:
        Tensor of the same shape with unit norm along `dim`.
    """
    return z / z.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity s(z1_i, z2_i) = zhat1_i . zhat2_i, in [-1, 1].

    Higher is more similar. Inputs are normalized here, so passing already-
    normalized embeddings is a no-op and passing raw ones is still correct.

    Args:
        z1: (N, D) embeddings.
        z2: (N, D) embeddings, paired row-wise with `z1`.

    Returns:
        (N,) similarities.
    """
    return (l2_normalize(z1) * l2_normalize(z2)).sum(dim=-1)


def euclidean_distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Row-wise Euclidean distance d(z1_i, z2_i) = ||zhat1_i - zhat2_i||_2 >= 0.

    Lower is more similar. On normalized embeddings d lies in [0, 2].

    Args:
        z1: (N, D) embeddings.
        z2: (N, D) embeddings, paired row-wise with `z1`.

    Returns:
        (N,) distances.
    """
    return (l2_normalize(z1) - l2_normalize(z2)).norm(p=2, dim=-1)


def cosine_similarity_matrix(z: torch.Tensor, other: torch.Tensor | None = None) -> torch.Tensor:
    """Full similarity matrix S = Zhat Zhat^T -- the matrix in poster panel 5, step 4.

    This is the object InfoNCE takes a row-wise softmax over, and the one dumped
    as the V3 identity-ordered heatmap.

    Args:
        z: (N, D) embeddings.
        other: optional (M, D) embeddings. Defaults to `z` (the N x N self-similarity
            case). Used with the gathered global batch in DDP (plan §6c).

    Returns:
        (N, M) similarities in [-1, 1]; (N, N) when `other` is None.
    """
    a = l2_normalize(z)
    b = a if other is None else l2_normalize(other)
    return a @ b.T


def euclidean_distance_matrix(z: torch.Tensor, other: torch.Tensor | None = None) -> torch.Tensor:
    """Full pairwise distance matrix on the unit sphere, via d^2 = 2 - 2s.

    Computed from the similarity matrix rather than by expanding differences:
    it is the same value analytically, one matmul instead of an (N, M, D)
    intermediate, and it keeps distances exactly consistent with
    `cosine_similarity_matrix` -- which matters because the miners compare
    d_ap against d_an at a 1e-5 tolerance.

    The clamp handles the small negative values (order -1e-7) that floating
    point produces on the diagonal, where s should be exactly 1; sqrt of a
    negative would yield NaN and poison every downstream loss.

    Args:
        z: (N, D) embeddings.
        other: optional (M, D) embeddings. Defaults to `z`.

    Returns:
        (N, M) distances in [0, 2]; (N, N) when `other` is None.
    """
    s = cosine_similarity_matrix(z, other)
    return (2.0 - 2.0 * s).clamp_min(0.0).sqrt()


def distance_from_similarity(s: torch.Tensor) -> torch.Tensor:
    """Convert cosine similarity to Euclidean distance: d = sqrt(2 - 2s).

    Valid only for L2-normalized embeddings. Exposed so reports and figures can
    move between the poster's two boxes without recomputing embeddings.
    """
    return (2.0 - 2.0 * s).clamp_min(0.0).sqrt()


def similarity_from_distance(d: torch.Tensor) -> torch.Tensor:
    """Convert Euclidean distance to cosine similarity: s = 1 - d^2 / 2.

    The inverse of `distance_from_similarity`, same normalized-input caveat.
    """
    return 1.0 - 0.5 * d.pow(2)
