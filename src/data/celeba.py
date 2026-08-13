"""CelebA loading, filtered by identity split (plan §3).

Two dataset classes, because training and evaluation ask different questions:

  * `CelebAIdentityDataset` yields `(image, identity)` and is what the PK sampler
    indexes into. Identity labels are remapped to a contiguous 0..C-1 range.
  * `CelebAPairDataset` yields images for a *named list of files* -- the fixed
    verification pair lists. It deduplicates: the 12,000 test pairs reference far
    fewer than 24,000 distinct images, and encoding a file twice is wasted GPU.

Both refuse to touch an image whose identity is outside the requested split, so
an identity leak cannot enter through the loader even if a split file were wrong.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_CELEBA_ROOT = Path("data/celeba")
IMAGE_DIR = "img_align_celeba"
IDENTITY_FILE = "identity_CelebA.txt"


def _require_celeba(root: Path) -> None:
    """Fail with manual-download instructions rather than silently degrading (plan §14.6)."""
    if not (root / IDENTITY_FILE).exists() or not (root / IMAGE_DIR).is_dir():
        raise FileNotFoundError(
            f"CelebA not found under {root}.\n\n"
            "Expected:\n"
            f"    {root}/{IMAGE_DIR}/   (202,599 jpgs)\n"
            f"    {root}/{IDENTITY_FILE}\n\n"
            "Auto-download: torchvision.datasets.CelebA(root='data', download=True)\n"
            "  (requires `pip install gdown`)\n"
            "Manual fallback: the Kaggle mirror of 'CelebFaces Attributes (CelebA) Dataset'.\n"
            "Do NOT substitute another face dataset."
        )


def read_identity_map(root: Path = DEFAULT_CELEBA_ROOT) -> dict[str, int]:
    """Return {filename: identity} for all 202,599 CelebA images."""
    _require_celeba(root)
    mapping: dict[str, int] = {}
    with (root / IDENTITY_FILE).open() as f:
        for raw in f:
            line = raw.strip()
            if line:
                filename, identity = line.split()
                mapping[filename] = int(identity)
    return mapping


def files_by_identity(root: Path = DEFAULT_CELEBA_ROOT) -> dict[int, list[str]]:
    """Return {identity: [filenames]}, filenames sorted for determinism."""
    grouped: dict[int, list[str]] = collections.defaultdict(list)
    for filename, identity in read_identity_map(root).items():
        grouped[identity].append(filename)
    return {i: sorted(f) for i, f in grouped.items()}


class CelebAIdentityDataset(Dataset):
    """CelebA images restricted to one identity split, yielding `(image, label)`.

    Labels are remapped to contiguous 0..C-1 indices. The original CelebA
    identity for row `i` is `self.identities[i]`; `self.labels` holds the
    remapped values and is what `PKSampler` should be constructed from.
    """

    def __init__(
        self,
        identities: frozenset[int] | set[int] | list[int],
        transform,
        root: Path = DEFAULT_CELEBA_ROOT,
        max_identities: int | None = None,
    ) -> None:
        """
        Args:
            identities: the CelebA identity ids to include (one split).
            transform: preprocessing pipeline from `build_transform`.
            root: CelebA root directory.
            max_identities: keep only the first N identities by id. Used for the
                E0 debug set (plan §3: "the first 10 CelebA train identities").
        """
        self.root = Path(root)
        self.transform = transform
        self.image_dir = self.root / IMAGE_DIR

        grouped = files_by_identity(self.root)
        wanted = sorted(set(identities) & set(grouped))
        if max_identities is not None:
            wanted = wanted[:max_identities]
        if not wanted:
            raise ValueError("no identities from the requested split are present in CelebA")

        self.files: list[str] = []
        self.identities: list[int] = []
        for identity in wanted:
            for filename in grouped[identity]:
                self.files.append(filename)
                self.identities.append(identity)

        # contiguous remap: PKSampler and the losses only care about equality
        self.identity_to_label = {identity: i for i, identity in enumerate(wanted)}
        self.labels = np.array(
            [self.identity_to_label[i] for i in self.identities], dtype=np.int64
        )
        self.n_identities = len(wanted)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_dir / self.files[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])

    def summary(self) -> str:
        counts = np.bincount(self.labels)
        return (
            f"{len(self):,} images / {self.n_identities:,} identities "
            f"(min {counts.min()}, median {int(np.median(counts))}, max {counts.max()} img/id)"
        )


class CelebAPairDataset(Dataset):
    """Deduplicated images for a verification pair list, yielding `(image, row)`.

    Encode once, then index the resulting embedding matrix with `pair_indices`
    to score pairs. The 12,000 test pairs reference ~15k distinct files, so
    deduplication saves roughly a third of the forward passes.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        transform,
        root: Path = DEFAULT_CELEBA_ROOT,
    ) -> None:
        self.root = Path(root)
        _require_celeba(self.root)
        self.image_dir = self.root / IMAGE_DIR
        self.transform = transform

        unique = sorted({f for pair in pairs for f in pair})
        self.files = unique
        row_of = {f: i for i, f in enumerate(unique)}
        self.pair_indices = np.array([[row_of[a], row_of[b]] for a, b in pairs], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_dir / self.files[index]).convert("RGB")
        return self.transform(image), index
