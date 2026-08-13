"""Miner tests (plan §12, Phase 2 gate item 4).

The gate assertion: on a crafted batch, every negative returned by the semi-hard
miner satisfies `d_ap < d_an < d_ap + alpha`.
"""

from __future__ import annotations

import pytest
import torch

from src.losses.geometry import euclidean_distance_matrix, l2_normalize
from src.losses.miners import MINERS, mine


@pytest.fixture(autouse=True)
def _determinism() -> None:
    torch.manual_seed(0)


def _pk_batch(p: int = 8, k: int = 4, d: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    labels = torch.arange(p).repeat_interleave(k)
    z = l2_normalize(torch.randn(p * k, d))
    return z, labels


# --- shared invariants --------------------------------------------------------


@pytest.mark.parametrize("miner", MINERS)
def test_returns_valid_triplets(miner: str) -> None:
    """Positive shares the anchor's label, negative does not, and never self."""
    z, labels = _pk_batch()
    a, p, n = mine(z, labels, miner, margin=0.2)

    assert len(a) == len(p) == len(n) == len(labels)
    assert (labels[a] == labels[p]).all(), "positive has a different identity"
    assert (labels[a] != labels[n]).all(), "negative shares the anchor's identity"
    assert (a != p).all(), "anchor paired with itself as its own positive"


@pytest.mark.parametrize("miner", MINERS)
def test_anchors_without_positives_are_dropped(miner: str) -> None:
    """A singleton identity cannot form a triplet and must not be emitted."""
    labels = torch.tensor([0, 0, 1, 1, 2])  # identity 2 is alone
    z = l2_normalize(torch.randn(5, 8))
    a, _, _ = mine(z, labels, miner, margin=0.2)
    assert 4 not in a.tolist()
    assert len(a) == 4


@pytest.mark.parametrize("miner", MINERS)
def test_single_identity_batch_returns_empty(miner: str) -> None:
    """No negatives exist; must return empty rather than crash."""
    labels = torch.zeros(6, dtype=torch.long)
    z = l2_normalize(torch.randn(6, 8))
    a, p, n = mine(z, labels, miner, margin=0.2)
    assert len(a) == len(p) == len(n) == 0


def test_unknown_miner_rejected() -> None:
    z, labels = _pk_batch()
    with pytest.raises(ValueError, match="miner must be one of"):
        mine(z, labels, "hardest-ever")


# --- semi-hard: the gate assertion --------------------------------------------


def _crafted_semi_hard_batch():
    """A batch where every anchor genuinely has a semi-hard negative available.

    Two identities on a circle. Same-identity points sit close; the other
    identity is placed at a range of angles so some of its points land inside
    the (d_ap, d_ap + alpha) band and others fall outside it on both sides.
    """
    angles = torch.tensor(
        [
            0.00, 0.05,          # identity 0: d_ap is small
            0.60, 0.75, 1.20,    # identity 1: spread across / beyond the band
            3.00, 3.05,          # identity 2: far away (easy negatives)
        ]
    )
    z = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    return l2_normalize(z), labels


def test_semi_hard_returns_only_negatives_inside_the_band() -> None:
    """GATE (plan §12): every returned negative satisfies d_ap < d_an < d_ap + alpha.

    Checked only for anchors that actually have a semi-hard negative available;
    the documented fallback applies to the rest and is tested separately.
    """
    margin = 0.5
    z, labels = _crafted_semi_hard_batch()
    distances = euclidean_distance_matrix(z)

    a, p, n = mine(z, labels, "semi-hard", margin=margin)

    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool)
    negative_mask = ~same

    n_checked = 0
    for anchor, positive, negative in zip(a.tolist(), p.tolist(), n.tolist()):
        d_ap = distances[anchor, positive]
        band = negative_mask[anchor] & (distances[anchor] > d_ap) & (
            distances[anchor] < d_ap + margin
        )
        if not band.any():
            continue  # fallback case, covered by the next test
        d_an = distances[anchor, negative]
        assert d_ap < d_an < d_ap + margin, (
            f"anchor {anchor}: d_ap={d_ap:.4f} d_an={d_an:.4f} "
            f"outside ({d_ap:.4f}, {d_ap + margin:.4f})"
        )
        n_checked += 1

    assert n_checked > 0, "crafted batch produced no semi-hard cases to check"
    assert not eye[a, p].any()


def test_semi_hard_holds_on_many_random_batches() -> None:
    """The band property must hold on random data too, not just the crafted case."""
    margin = 0.2
    for seed in range(20):
        z, labels = _pk_batch(p=6, k=4, d=8, seed=seed)
        distances = euclidean_distance_matrix(z)
        a, p, n = mine(z, labels, "semi-hard", margin=margin)
        same = labels[:, None] == labels[None, :]

        for anchor, positive, negative in zip(a.tolist(), p.tolist(), n.tolist()):
            d_ap = distances[anchor, positive]
            band = (~same[anchor]) & (distances[anchor] > d_ap) & (
                distances[anchor] < d_ap + margin
            )
            if band.any():
                d_an = distances[anchor, negative]
                assert d_ap < d_an < d_ap + margin


def test_semi_hard_falls_back_to_hardest_violating_negative() -> None:
    """With an empty band, take the closest negative that still violates the margin."""
    # Identity 0 pair is far apart (large d_ap); identity 1 sits very close to
    # the anchor, so every negative is nearer than the positive -> band is empty.
    z = l2_normalize(
        torch.tensor([[1.0, 0.0], [-1.0, 0.05], [0.99, 0.14], [0.98, 0.20]])
    )
    labels = torch.tensor([0, 0, 1, 1])
    distances = euclidean_distance_matrix(z)

    a, p, n = mine(z, labels, "semi-hard", margin=0.2)
    anchor = a.tolist().index(0)
    d_ap = distances[0, p[anchor]]
    band = (labels != 0) & (distances[0] > d_ap) & (distances[0] < d_ap + 0.2)
    assert not band.any(), "test setup failed: band should be empty"

    negatives = distances[0].clone()
    negatives[labels == 0] = float("inf")
    assert n[anchor].item() == negatives.argmin().item()


# --- batch-hard ---------------------------------------------------------------


def test_batch_hard_picks_the_furthest_positive_and_closest_negative() -> None:
    """Definitionally: max d_ap, min d_an, per anchor."""
    z, labels = _pk_batch(p=5, k=4, d=8)
    distances = euclidean_distance_matrix(z)
    a, p, n = mine(z, labels, "batch-hard")

    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool)

    for anchor, positive, negative in zip(a.tolist(), p.tolist(), n.tolist()):
        pos_d = distances[anchor].clone()
        pos_d[~(same[anchor] & ~eye[anchor])] = float("-inf")
        assert positive == pos_d.argmax().item()

        neg_d = distances[anchor].clone()
        neg_d[same[anchor]] = float("inf")
        assert negative == neg_d.argmin().item()


def test_batch_hard_is_deterministic() -> None:
    """No RNG involved -- two calls must agree exactly."""
    z, labels = _pk_batch()
    first = mine(z, labels, "batch-hard")
    second = mine(z, labels, "batch-hard")
    for x, y in zip(first, second):
        assert torch.equal(x, y)


def test_batch_hard_gives_the_smallest_margin_of_the_three() -> None:
    """Hardest positive minus closest negative is the tightest gap by construction."""
    z, labels = _pk_batch(p=8, k=4, d=16)
    distances = euclidean_distance_matrix(z)

    gaps = {}
    for miner in MINERS:
        a, p, n = mine(z, labels, miner, margin=0.2)
        gaps[miner] = (distances[a, n] - distances[a, p]).mean().item()

    assert gaps["batch-hard"] <= gaps["random"]


# --- random -------------------------------------------------------------------


def test_random_miner_varies_with_the_generator() -> None:
    z, labels = _pk_batch(p=8, k=4, d=16)
    first = mine(z, labels, "random", generator=torch.Generator().manual_seed(0))
    second = mine(z, labels, "random", generator=torch.Generator().manual_seed(1))
    assert not (torch.equal(first[1], second[1]) and torch.equal(first[2], second[2]))


def test_random_miner_is_reproducible_with_the_same_seed() -> None:
    z, labels = _pk_batch()
    first = mine(z, labels, "random", generator=torch.Generator().manual_seed(7))
    second = mine(z, labels, "random", generator=torch.Generator().manual_seed(7))
    for x, y in zip(first, second):
        assert torch.equal(x, y)


def test_random_miner_covers_multiple_positives_over_many_draws() -> None:
    """K=4 means 3 candidate positives per anchor; sampling must reach more than one."""
    z, labels = _pk_batch(p=4, k=4, d=8)
    chosen = set()
    for seed in range(30):
        _, p, _ = mine(z, labels, "random", generator=torch.Generator().manual_seed(seed))
        chosen.add(p[0].item())
    assert len(chosen) > 1
