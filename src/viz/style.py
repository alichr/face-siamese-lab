"""Shared figure styling so V1-V7 read as one system (plan §9).

Colors follow the job the encoding does, not taste:

  * **Categorical** (identity: which loss, which miner) -- a fixed hue order,
    assigned by slot and never cycled. Cross-experiment plots stay within the
    first three slots, which are the ones validated for all-pairs separation
    under colour-vision deficiency.
  * **Diverging** (polarity) for the similarity matrix. Cosine similarity has a
    meaningful zero -- orthogonal embeddings -- so blue/red around a neutral gray
    midpoint reads correctly. A sequential ramp would hide the sign, and a
    rainbow map would invent structure that is not in the data.
  * **Sequential** (magnitude) for anything running low-to-high with no natural
    zero, e.g. training progress across checkpoints.

Every figure is written to PNG *and* PDF (plan §14.5).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots, in fixed assignment order.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Stable identity -> color mapping. Colour follows the entity, never its rank,
# so dropping a loss from a comparison must not repaint the survivors.
LOSS_COLORS = {
    "contrastive": CATEGORICAL[0],
    "triplet": CATEGORICAL[1],
    "infonce": CATEGORICAL[2],
    "infonce-supcon": CATEGORICAL[6],
}
MINER_COLORS = {
    "random": CATEGORICAL[0],
    "semi-hard": CATEGORICAL[1],
    "batch-hard": CATEGORICAL[2],
}

# Same-person vs different-person: two entities, two fixed slots.
POSITIVE_COLOR = CATEGORICAL[0]
NEGATIVE_COLOR = CATEGORICAL[1]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"
SURFACE = "#ffffff"

# Diverging map for cosine similarity in [-1, 1]: two poles, neutral gray middle.
SIMILARITY_CMAP = LinearSegmentedColormap.from_list(
    "similarity", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#8f2020"]
)
# Sequential single hue for magnitude.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "sequential", ["#cde2fb", "#3987e5", "#0d366b"]
)


def apply_style() -> None:
    """Install the shared rcParams: thin marks, recessive grid, text-colored labels."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "font.size": 9,
            "figure.dpi": 130,
        }
    )


def save(fig, out_dir: Path | str, name: str) -> list[Path]:
    """Save a figure as both PNG and PDF (plan §14.5).

    Returns:
        The written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in ("png", "pdf"):
        path = out_dir / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def color_for(name: str, index: int = 0) -> str:
    """Stable color for a named series, falling back to the slot order."""
    key = str(name).lower()
    for table in (LOSS_COLORS, MINER_COLORS):
        if key in table:
            return table[key]
    return CATEGORICAL[index % len(CATEGORICAL)]
