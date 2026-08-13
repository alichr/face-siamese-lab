"""PKSampler tests (plan §12, Phase 0 gate item 1).

The gate assertion: *every* batch is exactly P identities x K images.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.pk_sampler import PKSampler


def make_labels(n_identities: int = 50, images_per_id: int = 10) -> np.ndarray:
    """Balanced synthetic label array: `n_identities` ids with `images_per_id` images each."""
    return np.repeat(np.arange(n_identities), images_per_id)


def make_ragged_labels() -> np.ndarray:
    """Unbalanced labels, mirroring CelebA (min 1, median ~21 images per identity)."""
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 30, size=60)
    return np.repeat(np.arange(60), counts)


# --- the gate assertion -------------------------------------------------------


@pytest.mark.parametrize(("p", "k"), [(4, 2), (8, 4), (16, 4), (10, 3), (64, 4)])
def test_every_batch_is_exactly_p_ids_by_k_images(p: int, k: int) -> None:
    """GATE: every batch has exactly P identities, each contributing exactly K images."""
    labels = make_labels(n_identities=100, images_per_id=12)
    sampler = PKSampler(labels, p=p, k=k, seed=0)

    n_batches = 0
    for batch in sampler:
        assert len(batch) == p * k, f"batch size {len(batch)} != p*k = {p * k}"

        batch_labels = labels[np.asarray(batch)]
        uniq, counts = np.unique(batch_labels, return_counts=True)
        assert len(uniq) == p, f"{len(uniq)} distinct identities in batch, expected p={p}"
        assert set(counts.tolist()) == {k}, f"per-identity counts {sorted(counts)} != all {k}"
        n_batches += 1

    assert n_batches == len(sampler) > 0


def test_batch_size_attribute_matches_yielded_batches() -> None:
    sampler = PKSampler(make_labels(), p=8, k=4)
    assert sampler.batch_size == 32
    assert all(len(b) == sampler.batch_size for b in sampler)


def test_indices_are_valid_and_distinct_within_batch() -> None:
    """No index repeats inside a batch when every identity has >= K distinct images."""
    labels = make_labels(n_identities=40, images_per_id=10)
    for batch in PKSampler(labels, p=8, k=4, seed=3):
        assert len(set(batch)) == len(batch), "duplicate dataset index inside one batch"
        assert all(0 <= i < len(labels) for i in batch)


# --- shuffling ----------------------------------------------------------------


def test_epochs_differ_after_set_epoch() -> None:
    """Without reshuffling, every epoch would replay identical batches."""
    sampler = PKSampler(make_labels(), p=8, k=4, seed=0)
    sampler.set_epoch(0)
    epoch0 = [list(b) for b in sampler]
    sampler.set_epoch(1)
    epoch1 = [list(b) for b in sampler]
    assert epoch0 != epoch1


def test_same_seed_and_epoch_is_reproducible() -> None:
    labels = make_labels()
    a = PKSampler(labels, p=8, k=4, seed=7)
    b = PKSampler(labels, p=8, k=4, seed=7)
    a.set_epoch(2)
    b.set_epoch(2)
    assert [list(x) for x in a] == [list(x) for x in b]


def test_different_seeds_differ() -> None:
    labels = make_labels()
    a = PKSampler(labels, p=8, k=4, seed=0)
    b = PKSampler(labels, p=8, k=4, seed=1)
    assert [list(x) for x in a] != [list(x) for x in b]


def test_identity_appears_at_most_once_per_batch_across_epoch() -> None:
    """P-groups are drawn from a permutation, so an epoch covers distinct identities."""
    labels = make_labels(n_identities=64, images_per_id=8)
    sampler = PKSampler(labels, p=8, k=4, seed=0)
    seen: list[int] = []
    for batch in sampler:
        batch_ids = np.unique(labels[np.asarray(batch)])
        seen.extend(batch_ids.tolist())
    assert len(seen) == len(set(seen)), "an identity was reused within one epoch"


# --- identities with fewer than K images --------------------------------------


def test_identities_with_too_few_images_are_dropped_by_default() -> None:
    """A duplicated positive sits at distance exactly 0 and teaches nothing."""
    labels = np.array([0] * 5 + [1] * 5 + [2] * 2 + [3] * 5 + [4] * 5 + [5] * 1)
    sampler = PKSampler(labels, p=2, k=4, seed=0)

    assert sampler.n_dropped_identities == 2  # ids 2 and 5
    assert sampler.n_identities == 4
    for batch in sampler:
        assert not {2, 5} & set(labels[np.asarray(batch)].tolist())


def test_allow_replacement_keeps_small_identities() -> None:
    labels = np.array([0] * 5 + [1] * 5 + [2] * 2 + [3] * 5)
    sampler = PKSampler(labels, p=4, k=4, seed=0, allow_replacement=True)

    assert sampler.n_dropped_identities == 0
    assert sampler.n_identities == 4
    for batch in sampler:
        assert len(batch) == 16
        uniq, counts = np.unique(labels[np.asarray(batch)], return_counts=True)
        assert len(uniq) == 4 and set(counts.tolist()) == {4}


def test_ragged_labels_still_satisfy_the_gate() -> None:
    """CelebA-like unbalanced counts must not break the exact-P-by-K guarantee."""
    labels = make_ragged_labels()
    sampler = PKSampler(labels, p=8, k=4, seed=0)
    for batch in sampler:
        uniq, counts = np.unique(labels[np.asarray(batch)], return_counts=True)
        assert len(batch) == 32 and len(uniq) == 8 and set(counts.tolist()) == {4}


# --- configuration errors -----------------------------------------------------


def test_k_less_than_2_is_rejected() -> None:
    """K=1 means no in-batch positive exists -- every loss here would be undefined."""
    with pytest.raises(ValueError, match="no in-batch positives"):
        PKSampler(make_labels(), p=8, k=1)


@pytest.mark.parametrize(("p", "k"), [(0, 4), (-1, 4), (8, 0), (8, -2)])
def test_non_positive_p_or_k_rejected(p: int, k: int) -> None:
    with pytest.raises(ValueError):
        PKSampler(make_labels(), p=p, k=k)


def test_too_few_usable_identities_raises_with_actionable_message() -> None:
    labels = make_labels(n_identities=3, images_per_id=10)
    with pytest.raises(ValueError, match="need at least p="):
        PKSampler(labels, p=8, k=4)


def test_non_1d_labels_rejected() -> None:
    with pytest.raises(ValueError, match="1-D"):
        PKSampler(np.zeros((10, 2)), p=2, k=2)


# --- length -------------------------------------------------------------------


def test_default_length_is_one_pass_over_identities() -> None:
    sampler = PKSampler(make_labels(n_identities=100, images_per_id=8), p=8, k=4)
    assert len(sampler) == 100 // 8


def test_num_batches_override_is_honoured() -> None:
    """More batches than one identity-pass: the sampler reshuffles and continues."""
    sampler = PKSampler(make_labels(n_identities=20, images_per_id=8), p=8, k=4, num_batches=25)
    batches = list(sampler)
    assert len(batches) == 25 == len(sampler)
    assert all(len(b) == 32 for b in batches)
