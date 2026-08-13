"""V5 — training curves (poster panels 4-5).

Three panels, because they answer three different questions and share no y-scale.
Putting them on one axis with two scales would be the classic dual-axis mistake;
small multiples keep each honest.

  loss             is it optimizing at all? Comparable only WITHIN a fixed loss
                   and miner -- an adaptive miner re-hardens its sample every
                   step, so its loss magnitude is not a progress metric.
  active fraction  how much of each batch still produces gradient. The curve that
                   separates the miners in E3.
  gap              mean s_pos - mean s_neg. The quantity every loss here is
                   ultimately trying to grow, and the one metric that IS
                   comparable across losses.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.viz.style import CATEGORICAL, apply_style, save

PANELS = [
    ("loss", "training loss", CATEGORICAL[0]),
    ("active_fraction", "active triplet fraction", CATEGORICAL[1]),
    ("gap", "mean s_pos − mean s_neg", CATEGORICAL[2]),
]


def plot_training_curves(
    curves_csv: Path | str,
    out_dir: Path | str,
    name: str = "v5_training_curves",
    title: str = "Training curves",
) -> list[Path]:
    """Plot loss / active-fraction / gap against epoch, plus val AUC.

    Columns absent from the CSV are skipped: contrastive has no
    `active_fraction`, InfoNCE has no triplet statistics.
    """
    import matplotlib.pyplot as plt

    apply_style()
    frame = pd.read_csv(curves_csv)

    available = [p for p in PANELS if p[0] in frame.columns and frame[p[0]].notna().any()]
    if "val_auc" in frame.columns and frame["val_auc"].notna().any():
        available.append(("val_auc", "validation ROC-AUC", CATEGORICAL[6]))

    fig, axes = plt.subplots(
        1, len(available), figsize=(3.4 * len(available), 3.0), squeeze=False
    )

    for ax, (column, label, color) in zip(axes[0], available):
        series = frame[["epoch", column]].dropna()
        ax.plot(series["epoch"], series[column], color=color)
        ax.set_xlabel("epoch")
        ax.set_title(label, fontsize=9.5, pad=10)
        # Headroom so the end-label never collides with the panel title.
        ax.margins(y=0.16)

        # Direct-label the final value rather than every point.
        if len(series):
            last = series.iloc[-1]
            ax.annotate(
                f"{last[column]:.4g}",
                xy=(last["epoch"], last[column]),
                xytext=(-4, 6),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=color,
                fontweight="semibold",
            )

    fig.suptitle(title, fontsize=11, fontweight="semibold", y=1.04)
    fig.tight_layout()
    return save(fig, out_dir, name)


def plot_loss_components(
    curves_csv: Path | str,
    out_dir: Path | str,
    name: str = "v5b_loss_components",
    title: str = "Distance / similarity diagnostics",
) -> list[Path] | None:
    """Per-step diagnostics that share a y-scale, so they belong on one axis."""
    import matplotlib.pyplot as plt

    apply_style()
    frame = pd.read_csv(curves_csv)

    groups = [
        ("mean_d_pos", "mean d over positives"),
        ("mean_d_neg", "mean d over negatives"),
        ("mean_d_ap", "mean d(a,p)"),
        ("mean_d_an", "mean d(a,n)"),
        ("mean_s_pos", "mean s over positives"),
        ("mean_s_neg", "mean s over negatives"),
    ]
    present = [(c, lbl) for c, lbl in groups if c in frame.columns and frame[c].notna().any()]
    if not present:
        return None

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for index, (column, label) in enumerate(present):
        series = frame[["epoch", column]].dropna()
        color = CATEGORICAL[index % len(CATEGORICAL)]
        ax.plot(series["epoch"], series[column], color=color, label=label)
        if len(series):
            last = series.iloc[-1]
            ax.annotate(
                label,
                xy=(last["epoch"], last[column]),
                xytext=(-4, 4),
                textcoords="offset points",
                ha="right",
                fontsize=7.5,
                color=color,
                fontweight="semibold",
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("distance / similarity")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    return save(fig, out_dir, name)
