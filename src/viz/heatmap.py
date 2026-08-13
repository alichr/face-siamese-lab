"""V3 — batch similarity matrix S, ordered by identity (poster panel 5, step 4).

This literally recreates the panel-5 illustration from real embeddings. Rows and
columns are sorted by identity, so the K x K same-identity blocks land on the
diagonal: a working encoder shows a visibly block-diagonal S, and a collapsed one
shows a uniformly hot square.

The colour map is DIVERGING, not sequential. Cosine similarity has a meaningful
zero -- orthogonal embeddings -- so blue (dissimilar) through neutral gray (zero)
to red (similar) puts the sign where a reader can see it. A sequential ramp would
render s = -0.8 and s = 0.0 as merely "two shades of light", losing the
distinction between "actively opposed" and "unrelated".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.viz.style import SIMILARITY_CMAP, TEXT_SECONDARY, apply_style, save


def plot_similarity_heatmap(
    embeddings: torch.Tensor,
    labels,
    out_dir: Path | str,
    name: str = "v3_similarity_heatmap",
    title: str = "Batch similarity matrix S (ordered by identity)",
    max_rows: int = 128,
) -> list[Path]:
    """Plot the identity-ordered N x N cosine similarity matrix.

    Args:
        embeddings: (N, d) embeddings.
        labels: (N,) identity labels.
        out_dir: figures directory.
        name: file stem.
        title: figure title.
        max_rows: cap on displayed rows -- a 256x256 grid of cells is unreadable
            in a report figure, and the block structure is already clear at 128.

    Returns:
        Written paths.
    """
    import matplotlib.pyplot as plt

    from src.losses.geometry import cosine_similarity_matrix

    apply_style()
    labels = np.asarray(labels)

    order = np.argsort(labels, kind="stable")[:max_rows]
    z = embeddings[order]
    ordered_labels = labels[order]
    similarity = cosine_similarity_matrix(z).numpy()

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    image = ax.imshow(similarity, cmap=SIMILARITY_CMAP, vmin=-1.0, vmax=1.0, interpolation="nearest")

    # Identity boundaries, so the K x K blocks are unambiguous.
    boundaries = np.flatnonzero(np.diff(ordered_labels)) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="#ffffff", linewidth=0.5, alpha=0.6)
        ax.axvline(boundary, color="#ffffff", linewidth=0.5, alpha=0.6)

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("cosine similarity", fontsize=8, color=TEXT_SECONDARY)
    bar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)

    ax.set_title(title)
    ax.set_xlabel(f"batch sample (first {len(order)}, grouped by identity)")
    ax.set_ylabel("batch sample")
    ax.grid(visible=False)

    return save(fig, out_dir, name)
