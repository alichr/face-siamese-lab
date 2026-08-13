"""Figure smoke tests.

These do not check that a figure looks good -- that needs eyes. They check that
every figure function runs and writes both PNG and PDF, because Phase 5 launches
~35 runs and a plotting crash in the last stage of a run would waste the whole job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.viz.curves import plot_loss_components, plot_training_curves
from src.viz.heatmap import plot_similarity_heatmap
from src.viz.histograms import plot_similarity_histograms
from src.viz.roc import plot_roc
from src.viz.style import CATEGORICAL, LOSS_COLORS, color_for
from src.viz.sweep_plots import plot_align_uniform, plot_sweep


def _scores(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = np.concatenate([rng.normal(0.7, 0.15, n), rng.normal(0.1, 0.15, n)])
    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    return scores, labels


def _assert_both_formats(paths) -> None:
    suffixes = {p.suffix for p in paths}
    assert suffixes == {".png", ".pdf"}, f"expected PNG+PDF, got {suffixes}"
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_v1_histograms(tmp_path) -> None:
    scores, labels = _scores()
    _assert_both_formats(
        plot_similarity_histograms(scores, labels, tmp_path, threshold=0.4, subtitle="AUC 0.99")
    )


def test_v1_handles_missing_threshold(tmp_path) -> None:
    scores, labels = _scores()
    _assert_both_formats(plot_similarity_histograms(scores, labels, tmp_path))


def test_v2_roc_single_and_multi(tmp_path) -> None:
    a, labels = _scores(seed=0)
    b, _ = _scores(seed=1)
    _assert_both_formats(plot_roc({"one": (a, labels)}, tmp_path, name="roc1"))
    _assert_both_formats(
        plot_roc({"infonce": (a, labels), "triplet": (b, labels)}, tmp_path, name="roc2")
    )


def test_v3_heatmap(tmp_path) -> None:
    torch.manual_seed(0)
    labels = np.repeat(np.arange(8), 4)
    z = torch.nn.functional.normalize(torch.randn(32, 16), dim=-1)
    _assert_both_formats(plot_similarity_heatmap(z, labels, tmp_path))


def test_v5_curves_with_partial_columns(tmp_path) -> None:
    """Contrastive has no active_fraction; InfoNCE has no triplet stats.
    Missing columns must be skipped, not crash."""
    frame = pd.DataFrame(
        {"epoch": range(10), "loss": np.linspace(5, 1, 10), "val_auc": np.linspace(0.6, 0.98, 10)}
    )
    csv = tmp_path / "curves.csv"
    frame.to_csv(csv, index=False)
    _assert_both_formats(plot_training_curves(csv, tmp_path))


def test_v5_curves_with_all_columns(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "epoch": range(10),
            "loss": np.linspace(5, 1, 10),
            "active_fraction": np.linspace(1.0, 0.3, 10),
            "gap": np.linspace(0.1, 0.6, 10),
            "val_auc": np.linspace(0.6, 0.98, 10),
            "mean_d_pos": np.linspace(1.2, 0.4, 10),
            "mean_d_neg": np.linspace(1.3, 1.4, 10),
        }
    )
    csv = tmp_path / "curves.csv"
    frame.to_csv(csv, index=False)
    _assert_both_formats(plot_training_curves(csv, tmp_path))
    _assert_both_formats(plot_loss_components(csv, tmp_path))


def test_v5b_returns_none_when_no_diagnostics(tmp_path) -> None:
    csv = tmp_path / "curves.csv"
    pd.DataFrame({"epoch": range(3), "loss": [3, 2, 1]}).to_csv(csv, index=False)
    assert plot_loss_components(csv, tmp_path) is None


def test_v6_align_uniform(tmp_path) -> None:
    points = {"contrastive": (0.8, -2.1), "triplet": (0.6, -2.6), "infonce": (0.5, -2.9)}
    _assert_both_formats(plot_align_uniform(points, tmp_path))


def test_v7_sweep_linear_and_log(tmp_path) -> None:
    _assert_both_formats(
        plot_sweep(
            [0.03, 0.05, 0.07, 0.1, 0.2],
            {"infonce": [0.88, 0.90, 0.91, 0.90, 0.86]},
            tmp_path,
            name="sweep_tau",
            x_label="temperature τ",
            y_label="LFW accuracy",
            title="E4",
            log_x=True,
        )
    )
    _assert_both_formats(
        plot_sweep(
            [64, 128, 256],
            {"a": [0.8, 0.85, 0.88], "b": [0.7, 0.8, 0.84]},
            tmp_path,
            name="sweep_batch",
            x_label="batch",
            y_label="acc",
            title="E5",
        )
    )


# --- palette rules ------------------------------------------------------------


def test_colour_follows_the_entity_not_the_rank() -> None:
    """Dropping a series from a comparison must not repaint the survivors."""
    assert color_for("infonce", 0) == color_for("infonce", 5) == LOSS_COLORS["infonce"]
    assert color_for("triplet", 9) == LOSS_COLORS["triplet"]


def test_unknown_series_falls_back_to_slot_order() -> None:
    assert color_for("e7_strong_aug", 0) == CATEGORICAL[0]
    assert color_for("e7_basic_aug", 1) == CATEGORICAL[1]


@pytest.mark.parametrize("name", ["contrastive", "triplet", "infonce"])
def test_the_three_losses_have_distinct_stable_colours(name: str) -> None:
    assert LOSS_COLORS[name] in CATEGORICAL[:3]
    assert len(set(LOSS_COLORS[n] for n in ("contrastive", "triplet", "infonce"))) == 3
