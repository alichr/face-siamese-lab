"""Geometry tests (plan §12).

The headline assertion is d^2 = 2 - 2s at atol 1e-5 -- the identity that makes the
poster's cosine-similarity and Euclidean-distance boxes interchangeable.
"""

from __future__ import annotations

import pytest
import torch

from src.losses.geometry import (
    cosine_similarity,
    cosine_similarity_matrix,
    distance_from_similarity,
    euclidean_distance,
    euclidean_distance_matrix,
    l2_normalize,
    similarity_from_distance,
)

ATOL = 1e-5


@pytest.fixture(autouse=True)
def _determinism() -> None:
    """Full determinism inside unit tests (plan §2)."""
    torch.manual_seed(0)


def _random_normalized(n: int = 256, d: int = 128) -> torch.Tensor:
    return l2_normalize(torch.randn(n, d, dtype=torch.float64))


def test_l2_normalize_gives_unit_norm() -> None:
    z = torch.randn(64, 128, dtype=torch.float64) * 37.0  # arbitrary scale
    norms = l2_normalize(z).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=ATOL)


def test_l2_normalize_survives_zero_vector() -> None:
    """A collapsed embedding must not produce NaN/inf (it would poison every loss)."""
    z = torch.zeros(4, 8, dtype=torch.float64)
    out = l2_normalize(z)
    assert torch.isfinite(out).all()


def test_d_squared_equals_2_minus_2s_rowwise() -> None:
    """THE identity, on random normalized pairs (plan §12)."""
    z1, z2 = _random_normalized(), _random_normalized()
    s = cosine_similarity(z1, z2)
    d = euclidean_distance(z1, z2)
    assert torch.allclose(d.pow(2), 2.0 - 2.0 * s, atol=ATOL)


def test_d_squared_equals_2_minus_2s_matrix() -> None:
    """Same identity for the full N x N matrices used by InfoNCE and the miners."""
    z = _random_normalized(n=64)
    s = cosine_similarity_matrix(z)
    d = euclidean_distance_matrix(z)
    assert torch.allclose(d.pow(2), 2.0 - 2.0 * s, atol=ATOL)


def test_matrix_matches_rowwise() -> None:
    """The matrix path and the row-wise path must agree; the miners rely on it."""
    z = _random_normalized(n=32)
    s_mat = cosine_similarity_matrix(z)
    idx = torch.arange(32)
    shifted = (idx + 7) % 32
    assert torch.allclose(s_mat[idx, shifted], cosine_similarity(z[idx], z[shifted]), atol=ATOL)


def test_similarity_and_distance_ranges() -> None:
    """s in [-1, 1] and d in [0, 2] on the unit sphere."""
    z = _random_normalized(n=64)
    s = cosine_similarity_matrix(z)
    d = euclidean_distance_matrix(z)
    assert s.min() >= -1.0 - ATOL and s.max() <= 1.0 + ATOL
    assert d.min() >= 0.0 and d.max() <= 2.0 + ATOL


def test_self_similarity_is_one_and_self_distance_is_zero() -> None:
    """The diagonal InfoNCE masks out: s_ii = 1, d_ii = 0 with no NaN from sqrt."""
    z = _random_normalized(n=64)
    diag_s = cosine_similarity_matrix(z).diagonal()
    diag_d = euclidean_distance_matrix(z).diagonal()
    assert torch.allclose(diag_s, torch.ones_like(diag_s), atol=ATOL)
    assert torch.allclose(diag_d, torch.zeros_like(diag_d), atol=ATOL)
    assert torch.isfinite(diag_d).all()


def test_distance_is_monotone_decreasing_in_similarity() -> None:
    """Why thresholding s and thresholding d are the same decision (poster panel 6)."""
    z = _random_normalized(n=48)
    s = cosine_similarity_matrix(z).flatten()
    d = euclidean_distance_matrix(z).flatten()
    order = torch.argsort(s, descending=True)
    d_sorted = d[order]
    assert (d_sorted[1:] - d_sorted[:-1] >= -ATOL).all()


def test_conversions_round_trip() -> None:
    z = _random_normalized(n=32)
    s = cosine_similarity_matrix(z)
    assert torch.allclose(distance_from_similarity(s).pow(2), 2.0 - 2.0 * s, atol=ATOL)
    assert torch.allclose(similarity_from_distance(distance_from_similarity(s)), s, atol=ATOL)


@pytest.mark.parametrize(
    ("s_expected", "d_expected"),
    [(1.0, 0.0), (0.0, 2.0**0.5), (-1.0, 2.0)],
)
def test_known_geometry_anchors(s_expected: float, d_expected: float) -> None:
    """Identical / orthogonal / antipodal -- the three points worth knowing by heart."""
    assert distance_from_similarity(torch.tensor(s_expected)).item() == pytest.approx(
        d_expected, abs=ATOL
    )


def test_float32_identity_holds_at_tolerance() -> None:
    """Training runs in fp32/bf16, not fp64 -- the identity must hold there too."""
    z1 = l2_normalize(torch.randn(512, 128))
    z2 = l2_normalize(torch.randn(512, 128))
    s = cosine_similarity(z1, z2)
    d = euclidean_distance(z1, z2)
    assert torch.allclose(d.pow(2), 2.0 - 2.0 * s, atol=ATOL)
