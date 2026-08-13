"""Identity split loading and the startup disjointness assertion (plan §3, §14.4).

Splits here are **identity-disjoint**, not image-disjoint, and the distinction is
the whole reason this module asserts rather than trusts.

Under an image-disjoint split the same person appears in both train and test.
Verification then asks "are these two faces the same person?" about identities
the encoder was explicitly trained to place at a specific spot on the sphere --
it can answer from memorized identity, not from a transferable notion of facial
similarity. The reported TAR@FAR would measure recall of training identities and
would not survive contact with a new face. Every headline number in this lab is
about *unseen* identities, so the split is load-bearing and is re-verified at the
startup of every run.

The generator is `scripts/make_splits.py`; this module only reads and checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SPLITS_DIR = Path("data/splits")

_SPLIT_FILES = {
    "train": "train_identities.txt",
    "val": "val_identities.txt",
    "test": "test_identities.txt",
}

PAIR_FILES = {
    "test": "internal_eval_pairs.txt",  # final benchmark, scored once per run
    "val": "internal_val_pairs.txt",  # per-epoch monitoring + checkpoint selection
}
EVAL_PAIRS_FILE = PAIR_FILES["test"]


@dataclass(frozen=True)
class IdentitySplits:
    """The three identity sets, guaranteed pairwise disjoint by construction."""

    train: frozenset[int]
    val: frozenset[int]
    test: frozenset[int]

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def summary(self) -> str:
        c = self.counts()
        total = sum(c.values())
        return (
            f"identity splits: train={c['train']:,} val={c['val']:,} "
            f"test={c['test']:,} (total {total:,}, pairwise disjoint)"
        )


def _read_identity_file(path: Path) -> frozenset[int]:
    if not path.exists():
        raise FileNotFoundError(
            f"split file missing: {path}\nGenerate it with:\n"
            f"    .venv/bin/python scripts/make_splits.py"
        )
    with path.open() as f:
        ids = [int(line) for line in (raw.strip() for raw in f) if line]
    if not ids:
        raise ValueError(f"split file is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"split file contains duplicate identities: {path}")
    return frozenset(ids)


def load_splits(splits_dir: Path | str = DEFAULT_SPLITS_DIR) -> IdentitySplits:
    """Read the three committed identity files and assert pairwise disjointness.

    Args:
        splits_dir: directory holding `{train,val,test}_identities.txt`.

    Returns:
        The three identity sets.

    Raises:
        FileNotFoundError: if a split file is missing.
        ValueError: if a file is empty, has duplicates, or the sets overlap.
    """
    d = Path(splits_dir)
    splits = IdentitySplits(
        train=_read_identity_file(d / _SPLIT_FILES["train"]),
        val=_read_identity_file(d / _SPLIT_FILES["val"]),
        test=_read_identity_file(d / _SPLIT_FILES["test"]),
    )
    assert_identity_disjoint(splits)
    return splits


def assert_identity_disjoint(splits: IdentitySplits) -> None:
    """Fail loudly if any two identity sets intersect (plan §14.4).

    Called at the startup of every run. A leak here silently inflates every
    verification metric downstream, and would be nearly impossible to spot from
    the metrics alone -- the numbers just look pleasingly good.

    Raises:
        ValueError: naming the overlapping pair and a sample of shared ids.
    """
    for a_name, b_name in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = getattr(splits, a_name) & getattr(splits, b_name)
        if overlap:
            sample = sorted(overlap)[:10]
            raise ValueError(
                f"IDENTITY LEAK: {a_name} and {b_name} share {len(overlap)} identities "
                f"(e.g. {sample}). Verification metrics would be measuring memorization, "
                f"not generalization -- regenerate with scripts/make_splits.py."
            )


def load_eval_pairs(
    splits_dir: Path | str = DEFAULT_SPLITS_DIR,
    which: str = "test",
) -> tuple[list[tuple[str, str]], list[int]]:
    """Load a fixed, seeded internal pair list (plan §8.1).

    The lists are committed so that every experiment in the matrix is scored on
    byte-identical pairs; regenerating them per run would make E1..E8 incomparable
    at the third decimal place for no reason.

    Args:
        splits_dir: directory holding the pair files.
        which: `"test"` for the final benchmark (test identities) or `"val"` for
            per-epoch monitoring and checkpoint selection (val identities).
            Selecting a checkpoint on the test list would bias every reported
            number, so the two pools are kept strictly separate.

    Returns:
        `(pairs, labels)` where pairs are `(filename_a, filename_b)` and labels
        are 1 for same-identity, 0 for different-identity.

    Raises:
        FileNotFoundError: if the pair file is missing.
        ValueError: on a malformed line.
    """
    if which not in PAIR_FILES:
        raise ValueError(f"which must be one of {sorted(PAIR_FILES)}, got {which!r}")

    path = Path(splits_dir) / PAIR_FILES[which]
    if not path.exists():
        raise FileNotFoundError(
            f"eval pair list missing: {path}\nGenerate it with:\n"
            f"    .venv/bin/python scripts/make_splits.py"
        )

    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{lineno}: expected 'file_a file_b label', got {line!r}")
            pairs.append((parts[0], parts[1]))
            labels.append(int(parts[2]))

    return pairs, labels
