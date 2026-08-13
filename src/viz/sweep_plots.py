"""V6 — alignment/uniformity scatter, and V7 — metric vs swept parameter.

V6 (plan §8.4, poster panel 7): each run is a point in the alignment-uniformity
plane. Both axes are "lower is better", so the lower-left corner is the good one.
The poster's claim that InfoNCE transfers better predicts it sits toward that
corner relative to the pairwise losses -- and the plane separates the two ways a
model can be bad, which no single accuracy number does.

V7: the swept parameter goes on x, the metric on y. Sweeps are ordered
magnitudes, so a line with markers is right; a bar chart would discard the
spacing information that makes a U-shape visible.
"""

from __future__ import annotations

from pathlib import Path

from src.viz.style import apply_style, color_for, save


def plot_align_uniform(
    points: dict[str, tuple[float, float]],
    out_dir: Path | str,
    name: str = "v6_align_uniform",
    title: str = "Alignment vs uniformity (lower-left is better)",
) -> list[Path]:
    """Scatter runs in the alignment-uniformity plane.

    Args:
        points: `{series_name: (alignment, uniformity)}`.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    for index, (series, (align, uniform)) in enumerate(points.items()):
        color = color_for(series, index)
        ax.scatter(
            align, uniform, s=90, color=color, edgecolors="#ffffff", linewidths=1.5, zorder=3
        )
        ax.annotate(
            series,
            xy=(align, uniform),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="semibold",
        )

    ax.set_xlabel("alignment  E‖ẑᵢ − ẑⱼ‖²  (lower = tighter positives)")
    ax.set_ylabel("uniformity  log E e^(−2‖ẑᵢ−ẑⱼ‖²)  (lower = better spread)")
    ax.set_title(title)
    return save(fig, out_dir, name)


def plot_sweep(
    x_values,
    series: dict[str, list[float]],
    out_dir: Path | str,
    name: str,
    x_label: str,
    y_label: str,
    title: str,
    log_x: bool = False,
) -> list[Path]:
    """V7: metric vs swept parameter, one line per series.

    Args:
        x_values: the swept parameter values.
        series: `{series_name: [metric per x]}`.
        log_x: log-scale the x axis (batch size, temperature).
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    for index, (label, values) in enumerate(series.items()):
        color = color_for(label, index)
        ax.plot(x_values, values, color=color, marker="o", label=label)
        ax.annotate(
            label,
            xy=(x_values[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="semibold",
            va="center",
        )

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(list(x_values))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    return save(fig, out_dir, name)
