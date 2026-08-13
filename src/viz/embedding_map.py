"""V4 — UMAP of embeddings for unseen identities (poster panel 1).

Panel 1 claims same-identity faces land close and different identities land far
apart. This is that claim, drawn -- and drawn on identities the encoder was never
trained on, which is the only version of the claim worth making.

Caveat worth stating in any report that shows it: UMAP is a non-linear projection
into 2D. Cluster *positions* and inter-cluster distances are not faithful to the
128-D geometry. What is meaningful is whether points of one colour group together
at all. Read it as a qualitative check on the quantitative metrics, never as
evidence on its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.viz.style import CATEGORICAL, apply_style, save


def plot_embedding_map(
    embeddings: torch.Tensor,
    labels,
    out_dir: Path | str,
    name: str = "v4_umap",
    n_identities: int = 30,
    title: str = "UMAP — 30 unseen identities",
    seed: int = 0,
) -> list[Path]:
    """Project embeddings to 2D with UMAP and colour by identity.

    Args:
        embeddings: (N, d) embeddings.
        labels: (N,) identity labels.
        out_dir: figures directory.
        name: file stem.
        n_identities: how many identities to draw.
        title: figure title.
        seed: UMAP random seed.

    Returns:
        Written paths.
    """
    import matplotlib.pyplot as plt
    import umap

    apply_style()
    labels = np.asarray(labels)

    chosen = np.unique(labels)[:n_identities]
    keep = np.isin(labels, chosen)
    z = embeddings[keep].numpy()
    y = labels[keep]

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
    projected = reducer.fit_transform(z)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for index, identity in enumerate(chosen):
        rows = y == identity
        ax.scatter(
            projected[rows, 0],
            projected[rows, 1],
            s=16,
            color=CATEGORICAL[index % len(CATEGORICAL)],
            edgecolors="#ffffff",
            linewidths=0.5,
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.text(
        0.0,
        1.02,
        f"{len(chosen)} identities, {int(keep.sum())} images · colour = identity · "
        "distances are not metric",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#52514e",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)

    return save(fig, out_dir, name)
