"""V1 — positive vs negative similarity histograms with the threshold (poster panel 6).

The single most informative figure in the lab. Panel 6 shows one accept/reject
decision at t=0.50; this shows the two score *distributions* that decision cuts
between, so the whole operating characteristic is visible at once.

The overlap region is the figure's point: it is exactly the set of pairs no
threshold can classify correctly, and its area is what the ROC curve integrates.
A threshold placed anywhere trades one tail for the other -- push t right and
false accepts fall while false rejects rise. A model that separates the two humps
completely has an AUC of 1 and no overlap; every real model here has some.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.viz.style import NEGATIVE_COLOR, POSITIVE_COLOR, TEXT_PRIMARY, apply_style, save


def plot_similarity_histograms(
    scores,
    labels,
    out_dir: Path | str,
    name: str = "v1_similarity_histograms",
    threshold: float | None = None,
    title: str = "Pair similarity distributions",
    subtitle: str | None = None,
) -> list[Path]:
    """Plot same-identity vs different-identity cosine similarity.

    Args:
        scores: (N,) cosine similarities.
        labels: (N,) 1 same-identity, 0 different.
        out_dir: figures directory.
        name: file stem.
        threshold: decision threshold to draw, if any.
        title: figure title.
        subtitle: optional line under the title (e.g. the headline metrics).

    Returns:
        Written paths.
    """
    import matplotlib.pyplot as plt

    apply_style()
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    positive = scores[labels == 1]
    negative = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    bins = np.linspace(min(scores.min(), -0.2), max(scores.max(), 1.0), 60)

    counts_neg, _, _ = ax.hist(negative, bins=bins, color=NEGATIVE_COLOR, alpha=0.75)
    counts_pos, _, _ = ax.hist(positive, bins=bins, color=POSITIVE_COLOR, alpha=0.75)

    # Headroom for the in-axes direct labels.
    ax.set_ylim(0, max(counts_neg.max(), counts_pos.max()) * 1.22)

    if threshold is not None:
        ax.axvline(threshold, color=TEXT_PRIMARY, linestyle="--", linewidth=1.4)
        ax.annotate(
            f"threshold {threshold:.3f}",
            xy=(threshold, ax.get_ylim()[1] * 0.62),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_PRIMARY,
            rotation=90,
            va="center",
        )

    # Direct labels sit ON each hump's peak, so identity never rests on colour
    # alone and nothing collides with the axis furniture below.
    for values, counts, color, label in (
        (negative, counts_neg, NEGATIVE_COLOR, "Different person"),
        (positive, counts_pos, POSITIVE_COLOR, "Same person"),
    ):
        if len(values) and counts.max() > 0:
            peak_bin = int(np.argmax(counts))
            peak_x = float((bins[peak_bin] + bins[peak_bin + 1]) / 2)
            ax.annotate(
                label,
                xy=(peak_x, counts.max()),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=color,
                fontweight="semibold",
            )

    ax.set_xlabel("cosine similarity  s(z₁, z₂)")
    ax.set_ylabel("pair count")
    ax.grid(axis="x", visible=False)

    # Title above, subtitle beneath it, neither overlapping the axes.
    ax.set_title(title, pad=22)
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=8,
            color="#52514e",
            va="bottom",
        )

    return save(fig, out_dir, name)
