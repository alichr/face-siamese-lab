"""Split tests — guards the identity-disjointness invariant (Phase 0 gate item 2).

An identity leak is the one bug in this project that makes results look *better*,
which is exactly why it gets a test rather than a comment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.splits import (
    DEFAULT_SPLITS_DIR,
    IdentitySplits,
    assert_identity_disjoint,
    load_eval_pairs,
    load_splits,
)

SPLITS_EXIST = (Path(DEFAULT_SPLITS_DIR) / "train_identities.txt").exists()
needs_splits = pytest.mark.skipif(
    not SPLITS_EXIST, reason="run scripts/make_splits.py first"
)


# --- the assertion itself (no data needed) ------------------------------------


def test_disjoint_splits_pass() -> None:
    assert_identity_disjoint(
        IdentitySplits(train=frozenset({1, 2}), val=frozenset({3}), test=frozenset({4}))
    )


@pytest.mark.parametrize(
    ("train", "val", "test", "expect"),
    [
        ({1, 2}, {2, 3}, {4}, "train and val"),
        ({1, 2}, {3}, {2, 4}, "train and test"),
        ({1}, {2, 3}, {3, 4}, "val and test"),
    ],
)
def test_overlapping_splits_raise(
    train: set[int], val: set[int], test: set[int], expect: str
) -> None:
    """The failure must name which pair leaked, not just that something is wrong."""
    splits = IdentitySplits(frozenset(train), frozenset(val), frozenset(test))
    with pytest.raises(ValueError, match="IDENTITY LEAK") as excinfo:
        assert_identity_disjoint(splits)
    assert expect in str(excinfo.value)


def test_leak_message_reports_count_and_examples() -> None:
    shared = set(range(100, 120))
    splits = IdentitySplits(frozenset(shared | {1}), frozenset(shared | {2}), frozenset({3}))
    with pytest.raises(ValueError, match=r"share 20 identities"):
        assert_identity_disjoint(splits)


def test_missing_split_file_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_splits.py"):
        load_splits(tmp_path)


def test_duplicate_identity_in_file_rejected(tmp_path: Path) -> None:
    for name in ("train", "val", "test"):
        (tmp_path / f"{name}_identities.txt").write_text("1\n2\n")
    (tmp_path / "train_identities.txt").write_text("1\n1\n2\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_splits(tmp_path)


# --- the committed splits (plan §3 targets) -----------------------------------


@needs_splits
def test_committed_splits_are_disjoint_and_correctly_sized() -> None:
    splits = load_splits()
    counts = splits.counts()
    assert counts["train"] == 8_000
    assert counts["val"] == 1_000
    assert sum(counts.values()) == 10_177  # every CelebA identity used exactly once


@needs_splits
def test_committed_eval_pairs_are_balanced_and_unique() -> None:
    pairs, labels = load_eval_pairs()
    assert sum(labels) == 6_000
    assert len(labels) - sum(labels) == 6_000
    assert len(set(pairs)) == len(pairs), "duplicate pair would be double-counted by the ROC"


@needs_splits
def test_val_pairs_are_balanced_and_come_from_val_identities_only() -> None:
    """Checkpoint selection must not see test identities (that would bias every number)."""
    pairs, labels = load_eval_pairs(which="val")
    assert sum(labels) == 3_000
    assert len(labels) - sum(labels) == 3_000
    assert len(set(pairs)) == len(pairs)


@needs_splits
def test_val_and_test_pair_lists_are_disjoint() -> None:
    """No image may appear in both the selection set and the benchmark set."""
    val_pairs, _ = load_eval_pairs(which="val")
    test_pairs, _ = load_eval_pairs(which="test")
    val_files = {f for pair in val_pairs for f in pair}
    test_files = {f for pair in test_pairs for f in pair}
    assert not (val_files & test_files)
    assert not (set(val_pairs) & set(test_pairs))


def test_unknown_pair_list_rejected() -> None:
    with pytest.raises(ValueError, match="which must be one of"):
        load_eval_pairs(which="train")


@needs_splits
def test_eval_pairs_never_reference_a_training_identity() -> None:
    """The internal benchmark must sit entirely on unseen identities (plan §8.1)."""
    import collections

    splits = load_splits()
    by_file: dict[str, int] = {}
    with open("data/celeba/identity_CelebA.txt") as f:
        for raw in f:
            filename, identity = raw.split()
            by_file[filename] = int(identity)

    pairs, labels = load_eval_pairs()
    used = {by_file[f] for pair in pairs for f in pair}
    assert not (used & splits.train), "eval pair references a TRAIN identity"
    assert not (used & splits.val), "eval pair references a VAL identity"
    assert used <= splits.test

    # positives really are same-identity, negatives really are cross-identity
    verdicts = collections.Counter(
        (by_file[a] == by_file[b], y) for (a, b), y in zip(pairs, labels)
    )
    assert verdicts[(True, 1)] == 6_000, "a 'positive' pair was not same-identity"
    assert verdicts[(False, 0)] == 6_000, "a 'negative' pair was not cross-identity"
