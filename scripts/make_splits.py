"""Generate the committed identity splits and internal eval pair list (plan §3, §8.1).

Run once; the outputs are committed and then treated as fixed for the whole
experiment matrix, so every run in E1..E8 is scored on byte-identical pairs.

    .venv/bin/python scripts/make_splits.py

Writes to `data/splits/`:
    train_identities.txt      8,000 CelebA identities
    val_identities.txt        1,000 identities
    test_identities.txt       the remaining ~1,177 identities
    internal_eval_pairs.txt   6,000 positive + 6,000 negative pairs, TEST ids only
    internal_val_pairs.txt    3,000 positive + 3,000 negative pairs, VAL ids only
    stats.md                  per-split image/identity counts (Phase 0 gate item 3)

The two pair lists exist for different jobs and must not be confused. The val
list drives per-epoch monitoring and best-checkpoint selection; the test list is
the final benchmark, touched once per run at the end. Selecting a checkpoint on
the test list would make every reported number optimistically biased -- the
model would be chosen for the very pairs it is then scored on.

Splits are by *identity*, never by image -- see `src/data/splits.py` for why.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from src.data.splits import IdentitySplits, assert_identity_disjoint

IDENTITY_FILE = "identity_CelebA.txt"


def read_celeba_identities(celeba_root: Path) -> dict[int, list[str]]:
    """Parse `identity_CelebA.txt` into {identity: [filenames]}.

    Raises:
        FileNotFoundError: with manual-download instructions (plan §14.6).
    """
    path = celeba_root / IDENTITY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n\n"
            "CelebA is missing. Download the aligned dataset (Kaggle mirror of\n"
            "'CelebFaces Attributes (CelebA) Dataset') and lay it out as:\n"
            f"    {celeba_root}/img_align_celeba/   (202,599 jpgs)\n"
            f"    {celeba_root}/identity_CelebA.txt\n"
            "Do NOT substitute another face dataset -- the whole matrix assumes CelebA."
        )

    by_identity: dict[int, list[str]] = collections.defaultdict(list)
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            filename, identity = line.split()
            by_identity[int(identity)].append(filename)
    return dict(by_identity)


def make_identity_splits(
    identities: list[int], n_train: int, n_val: int, seed: int
) -> IdentitySplits:
    """Shuffle identities once with a fixed seed and cut into train/val/test."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(sorted(identities)))

    if len(shuffled) < n_train + n_val + 1:
        raise ValueError(
            f"only {len(shuffled)} identities available, need > {n_train + n_val}"
        )

    splits = IdentitySplits(
        train=frozenset(int(i) for i in shuffled[:n_train]),
        val=frozenset(int(i) for i in shuffled[n_train : n_train + n_val]),
        test=frozenset(int(i) for i in shuffled[n_train + n_val :]),
    )
    assert_identity_disjoint(splits)
    return splits


def make_eval_pairs(
    by_identity: dict[int, list[str]],
    eval_ids: frozenset[int],
    n_per_class: int,
    seed: int,
) -> list[tuple[str, str, int]]:
    """Build `n_per_class` positive and `n_per_class` negative pairs from `eval_ids`.

    Positives are two distinct images of one identity; negatives are one image
    each from two distinct identities. Pairs are deduplicated, so the same
    unordered pair never appears twice and cannot be double-counted by the ROC.
    """
    rng = np.random.default_rng(seed)

    eligible = sorted(i for i in eval_ids if len(by_identity.get(i, [])) >= 2)
    if len(eligible) < 2:
        raise ValueError("need >= 2 eval identities with >= 2 images each")

    positives: set[tuple[str, str]] = set()
    guard = 0
    while len(positives) < n_per_class:
        identity = eligible[rng.integers(len(eligible))]
        files = by_identity[identity]
        a, b = rng.choice(len(files), size=2, replace=False)
        positives.add(tuple(sorted((files[a], files[b]))))
        guard += 1
        if guard > 200 * n_per_class:
            raise RuntimeError(
                f"only found {len(positives)} unique positive pairs of {n_per_class}"
            )

    all_eval_ids = sorted(i for i in eval_ids if by_identity.get(i))
    negatives: set[tuple[str, str]] = set()
    guard = 0
    while len(negatives) < n_per_class:
        i, j = rng.choice(len(all_eval_ids), size=2, replace=False)
        fa = by_identity[all_eval_ids[i]]
        fb = by_identity[all_eval_ids[j]]
        a = fa[rng.integers(len(fa))]
        b = fb[rng.integers(len(fb))]
        negatives.add(tuple(sorted((a, b))))
        guard += 1
        if guard > 200 * n_per_class:
            raise RuntimeError(
                f"only found {len(negatives)} unique negative pairs of {n_per_class}"
            )

    return [(a, b, 1) for a, b in sorted(positives)] + [(a, b, 0) for a, b in sorted(negatives)]


def write_stats(
    out_dir: Path, by_identity: dict[int, list[str]], splits: IdentitySplits
) -> str:
    """Write and return the per-split stats table (Phase 0 gate item 3)."""
    lines = [
        "# Split statistics",
        "",
        "Generated by `scripts/make_splits.py`. Splits are identity-disjoint (plan §3).",
        "",
        "| split | identities | images | min img/id | median img/id | max img/id |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("train", "val", "test"):
        ids = getattr(splits, name)
        counts = np.array([len(by_identity[i]) for i in sorted(ids) if i in by_identity])
        lines.append(
            f"| {name} | {len(ids):,} | {int(counts.sum()):,} | {int(counts.min())} | "
            f"{int(np.median(counts))} | {int(counts.max())} |"
        )

    total_ids = sum(len(getattr(splits, n)) for n in ("train", "val", "test"))
    lines += ["", f"Total identities: {total_ids:,} · total images: {sum(len(v) for v in by_identity.values()):,}"]

    text = "\n".join(lines) + "\n"
    (out_dir / "stats.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--celeba-root", type=Path, default=Path("data/celeba"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-eval-pairs", type=int, default=6000, help="test, per class")
    ap.add_argument("--n-val-pairs", type=int, default=3000, help="val, per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_identity = read_celeba_identities(args.celeba_root)
    print(
        f"CelebA: {sum(len(v) for v in by_identity.values()):,} images / "
        f"{len(by_identity):,} identities"
    )

    splits = make_identity_splits(sorted(by_identity), args.n_train, args.n_val, args.seed)
    for name in ("train", "val", "test"):
        ids = sorted(getattr(splits, name))
        (args.out_dir / f"{name}_identities.txt").write_text(
            "\n".join(str(i) for i in ids) + "\n"
        )
    print(splits.summary())

    # Test pairs: the final benchmark. Val pairs: per-epoch monitoring and
    # checkpoint selection. Different identity pools, different seeds, never mixed.
    for filename, ids, n_pairs, seed_offset, tag in (
        ("internal_eval_pairs.txt", splits.test, args.n_eval_pairs, 0, "test"),
        ("internal_val_pairs.txt", splits.val, args.n_val_pairs, 1, "val "),
    ):
        pairs = make_eval_pairs(by_identity, ids, n_pairs, args.seed + seed_offset)
        (args.out_dir / filename).write_text(
            "\n".join(f"{a} {b} {y}" for a, b, y in pairs) + "\n"
        )
        n_pos = sum(y for *_, y in pairs)
        print(
            f"internal {tag} pairs: {n_pos:,} positive + {len(pairs) - n_pos:,} negative "
            f"-> {filename}"
        )

    print()
    print(write_stats(args.out_dir, by_identity, splits))


if __name__ == "__main__":
    main()
