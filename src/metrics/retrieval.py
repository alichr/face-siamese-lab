"""Retrieval metrics: Recall@K on test identities (plan §8.3).

Verification asks a yes/no question about one pair. Retrieval asks a harder one:
given a query face, is the right identity in the top K of an entire gallery?
Recall@1 in particular degrades much faster than pair accuracy as the gallery
grows, because a single confusable impostor anywhere in the gallery can outrank
the true match. Reporting both is how the poster's "verification" and
"representation learning" claims get separated.

Protocol (plan §8.3): one gallery image per test identity, every remaining image
of those identities becomes a query.
"""

from __future__ import annotations

import numpy as np
import torch

from src.losses.geometry import l2_normalize


def recall_at_k(
    embeddings: torch.Tensor,
    labels: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    seed: int = 0,
) -> dict[str, float]:
    """Recall@K with one gallery image per identity.

    Args:
        embeddings: (N, d) embeddings for the evaluation images.
        labels: (N,) identity label per row.
        ks: the K values to report.
        seed: RNG seed for gallery selection.

    Returns:
        `recall@1`, `recall@5`, ... plus `n_queries` and `gallery_size`.

    Raises:
        ValueError: if fewer than 2 identities have 2+ images.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    gallery_rows: list[int] = []
    gallery_labels: list[int] = []
    query_rows: list[int] = []

    for identity in np.unique(labels):
        rows = np.flatnonzero(labels == identity)
        if len(rows) < 2:
            continue  # needs one gallery image and at least one query
        chosen = int(rng.choice(rows))
        gallery_rows.append(chosen)
        gallery_labels.append(int(identity))
        query_rows.extend(int(r) for r in rows if r != chosen)

    if len(gallery_rows) < 2:
        raise ValueError("need >= 2 identities with >= 2 images each for retrieval")

    gallery = l2_normalize(embeddings[gallery_rows])
    queries = l2_normalize(embeddings[query_rows])
    gallery_label_array = np.asarray(gallery_labels)
    query_label_array = labels[query_rows]

    similarity = queries @ gallery.T  # (Q, G)
    max_k = min(max(ks), similarity.shape[1])
    top = similarity.topk(max_k, dim=1).indices.numpy()
    retrieved = gallery_label_array[top]  # (Q, max_k)

    hits = retrieved == query_label_array[:, None]

    out: dict[str, float] = {}
    for k in ks:
        kk = min(k, max_k)
        out[f"recall@{k}"] = float(hits[:, :kk].any(axis=1).mean())

    out["n_queries"] = float(len(query_rows))
    out["gallery_size"] = float(len(gallery_rows))
    return out
