"""LFW pair loading, 10-fold protocol (plan §3, §8.2). Evaluation only, never training.

**Provenance note.** `torchvision.datasets.LFWPairs` no longer auto-downloads
(upstream raises "no longer available for automatic download"), and the canonical
host `vis-www.cs.umass.edu` does not resolve. Phase 0 fetched the identical
funneled dataset plus the official `pairs.txt` through
`sklearn.datasets.fetch_lfw_pairs`, which mirrors from figshare. The layout it
produces is what this module reads:

    data/lfw_sklearn/lfw_home/
        pairs.txt                       official 10-fold protocol file
        lfw_funneled/<Person_Name>/<Person_Name>_NNNN.jpg

`pairs.txt` format: a header line `10 300` (10 folds, 300 pairs per class per
fold), then for each fold 300 positive lines `name idx1 idx2` followed by 300
negative lines `name1 idx1 name2 idx2`. Fold membership is positional, so the
file order is load-bearing and must not be sorted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_LFW_ROOT = Path("data/lfw_sklearn/lfw_home")
IMAGE_DIR = "lfw_funneled"
PAIRS_FILE = "pairs.txt"


@dataclass(frozen=True)
class LFWPairs:
    """Parsed LFW pairs with fold assignment.

    Attributes:
        paths: `(file_a, file_b)` paths relative to the funneled image dir.
        labels: 1 for same-person, 0 for different.
        folds: fold index 0..9 for each pair -- the 10-fold protocol needs it.
    """

    paths: list[tuple[str, str]]
    labels: np.ndarray
    folds: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


def _image_name(person: str, index: int) -> str:
    return f"{person}/{person}_{int(index):04d}.jpg"


def load_lfw_pairs(root: Path | str = DEFAULT_LFW_ROOT) -> LFWPairs:
    """Parse the official `pairs.txt` into paths, labels and fold indices.

    Raises:
        FileNotFoundError: with manual instructions if LFW is absent.
        ValueError: if the file does not match the documented format.
    """
    root = Path(root)
    pairs_path = root / PAIRS_FILE
    if not pairs_path.exists() or not (root / IMAGE_DIR).is_dir():
        raise FileNotFoundError(
            f"LFW not found under {root}.\n\n"
            "torchvision can no longer download LFW and the UMass host is unreachable.\n"
            "Fetch it with scikit-learn instead:\n"
            "    from sklearn.datasets import fetch_lfw_pairs\n"
            "    fetch_lfw_pairs(subset='10_folds', funneled=True, color=True,\n"
            "                    resize=1.0, data_home='data/lfw_sklearn')\n"
        )

    lines = [ln.strip() for ln in pairs_path.read_text().splitlines() if ln.strip()]
    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(f"unexpected pairs.txt header: {lines[0]!r}")
    n_folds, per_class = int(header[0]), int(header[1])

    expected = n_folds * per_class * 2
    body = lines[1:]
    if len(body) != expected:
        raise ValueError(f"pairs.txt has {len(body)} pair lines, expected {expected}")

    paths: list[tuple[str, str]] = []
    labels: list[int] = []
    folds: list[int] = []

    cursor = 0
    for fold in range(n_folds):
        for is_positive in (True, False):
            for _ in range(per_class):
                parts = body[cursor].split()
                cursor += 1
                if is_positive:
                    if len(parts) != 3:
                        raise ValueError(f"bad positive line: {parts}")
                    person, i, j = parts
                    paths.append((_image_name(person, i), _image_name(person, j)))
                    labels.append(1)
                else:
                    if len(parts) != 4:
                        raise ValueError(f"bad negative line: {parts}")
                    person_a, i, person_b, j = parts
                    paths.append((_image_name(person_a, i), _image_name(person_b, j)))
                    labels.append(0)
                folds.append(fold)

    return LFWPairs(
        paths=paths,
        labels=np.asarray(labels, dtype=np.int64),
        folds=np.asarray(folds, dtype=np.int64),
    )


class LFWImageDataset(Dataset):
    """Deduplicated LFW images for a pair list, yielding `(image, row_index)`."""

    def __init__(
        self,
        pairs: LFWPairs,
        transform,
        root: Path | str = DEFAULT_LFW_ROOT,
    ) -> None:
        self.image_dir = Path(root) / IMAGE_DIR
        self.transform = transform

        unique = sorted({p for pair in pairs.paths for p in pair})
        self.files = unique
        row_of = {f: i for i, f in enumerate(unique)}
        self.pair_indices = np.array(
            [[row_of[a], row_of[b]] for a, b in pairs.paths], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_dir / self.files[index]).convert("RGB")
        return self.transform(image), index
