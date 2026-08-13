"""V2 — ROC curves on a log-FPR axis (poster panel 6 / Tips).

The x-axis is logarithmic on purpose. A linear ROC spends most of its width on
false-accept rates no verification system would ever operate at, and squeezes the
region that matters -- FAR between 1e-4 and 1e-2 -- into a sliver at the left
edge. Two models that differ by 15 points of TAR@FAR=1e-3 can look identical on a
linear plot. On a log axis that gap is the widest part of the figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.viz.style import TEXT_SECONDARY, apply_style, color_for, save

FAR_MARKERS = (1e-3, 1e-2)


def plot_roc(
    curves: dict[str, tuple],
    out_dir: Path | str,
    name: str = "v2_roc",
    title: str = "ROC — verification",
) -> list[Path]:
    """Plot one or more ROC curves with log-scaled FPR.

    Args:
        curves: `{series_name: (scores, labels)}`.
        out_dir: figures directory.
        name: file stem.
        title: figure title.

    Returns:
        Written paths.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))

    for index, (series, (scores, labels)) in enumerate(curves.items()):
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        far, tar, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        color = color_for(series, index)

        # Clip zero FAR so the log axis can show the leading point.
        far = np.clip(far, 1e-5, None)
        ax.plot(far, tar, color=color, label=f"{series}  (AUC {auc:.4f})")

        # Direct label at the right end of each curve.
        ax.annotate(
            series,
            xy=(far[-1], tar[-1]),
            xytext=(-4, -10),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=color,
            fontweight="semibold",
        )

    for marker in FAR_MARKERS:
        ax.axvline(marker, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.annotate(
            f"FAR {marker:g}",
            xy=(marker, 0.02),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=7,
            color=TEXT_SECONDARY,
            rotation=90,
        )

    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("false accept rate (log)")
    ax.set_ylabel("true accept rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return save(fig, out_dir, name)
