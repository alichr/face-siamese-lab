"""Embedding extraction and verification scoring (plan §8.1).

Phase 1 scope: embed a fixed pair list and report ROC-AUC. The full suite
(TAR@FAR, EER, LFW 10-fold, retrieval, alignment/uniformity) arrives in Phase 3
and will build on `embed_dataset` / `score_pairs` unchanged.

Evaluation always runs with the deterministic eval transform and the model in
eval mode, so a checkpoint scores identically every time it is measured. Without
that, two runs of the matrix could differ by more than the effect being studied.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.celeba import CelebAPairDataset
from src.data.transforms import build_transform
from src.losses.geometry import l2_normalize


@torch.no_grad()
def embed_dataset(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 8,
    amp_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Encode every row of `dataset`, returning embeddings in dataset order.

    The dataset must yield `(image, row_index)`; rows are scattered back by index
    rather than concatenated, so a shuffled or multi-worker loader cannot silently
    permute the embeddings relative to the pair list.

    Args:
        model: encoder; switched to eval mode and restored afterwards.
        dataset: yields `(image, row_index)`.
        device: compute device.
        batch_size: eval batch size.
        num_workers: dataloader workers.
        amp_dtype: autocast dtype, or None for full precision.

    Returns:
        (len(dataset), d) float32 embeddings on CPU.
    """
    was_training = model.training
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    out: torch.Tensor | None = None
    for images, rows in loader:
        images = images.to(device, non_blocking=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                z = model(images)
        else:
            z = model(images)
        z = z.float().cpu()
        if out is None:
            out = torch.empty(len(dataset), z.shape[1], dtype=torch.float32)
        out[rows] = z

    if was_training:
        model.train()

    assert out is not None, "empty dataset"
    return out


def score_pairs(embeddings: torch.Tensor, pair_indices: np.ndarray) -> np.ndarray:
    """Cosine similarity for each pair (poster panel 6: the score being thresholded).

    Args:
        embeddings: (M, d) embeddings.
        pair_indices: (P, 2) row indices into `embeddings`.

    Returns:
        (P,) cosine similarities in [-1, 1].
    """
    z = l2_normalize(embeddings)
    a = z[pair_indices[:, 0]]
    b = z[pair_indices[:, 1]]
    return (a * b).sum(dim=1).numpy()


@torch.no_grad()
def evaluate_pairs(
    model: torch.nn.Module,
    pairs: list[tuple[str, str]],
    labels: list[int],
    device: torch.device,
    celeba_root=None,
    batch_size: int = 512,
    num_workers: int = 8,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Embed a pair list and return verification metrics.

    Phase 1 returns ROC-AUC plus the similarity gap; Phase 3 extends this.

    Args:
        model: encoder.
        pairs: `(file_a, file_b)` filenames.
        labels: 1 for same-identity, 0 otherwise.
        device: compute device.
        celeba_root: CelebA root, or None for the default.
        batch_size: eval batch size.
        num_workers: dataloader workers.
        amp_dtype: autocast dtype, or None for full precision.

    Returns:
        Dict with `auc`, `mean_s_pos`, `mean_s_neg`, `gap`.
    """
    kwargs = {} if celeba_root is None else {"root": celeba_root}
    dataset = CelebAPairDataset(pairs, build_transform(train=False), **kwargs)

    embeddings = embed_dataset(
        model, dataset, device, batch_size=batch_size, num_workers=num_workers, amp_dtype=amp_dtype
    )
    scores = score_pairs(embeddings, dataset.pair_indices)
    y = np.asarray(labels)

    mean_pos = float(scores[y == 1].mean())
    mean_neg = float(scores[y == 0].mean())

    return {
        "auc": float(roc_auc_score(y, scores)),
        "mean_s_pos": mean_pos,
        "mean_s_neg": mean_neg,
        "gap": mean_pos - mean_neg,
    }
