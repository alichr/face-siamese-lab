"""Aggregate every run into one table plus cross-experiment figures (plan §9, §10).

    .venv/bin/python scripts/compare.py

Writes to `results/comparison/`:
    master_table.md / .csv   every run, one row
    v2_roc_e1.{png,pdf}      ROC overlay for the E1 loss face-off
    v6_align_uniform.{png,pdf}
    v7_<experiment>.{png,pdf}  one sweep plot per experiment

Figures are drawn from `metrics.json` only, never from a run's own figures, so
the comparison cannot silently disagree with the per-run reports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.viz.style import apply_style, save
from src.viz.sweep_plots import plot_align_uniform, plot_sweep

RESULTS = Path("results")
OUT = RESULTS / "comparison"

# experiment -> (swept parameter label, extractor from the run name, log x?)
SWEEPS: dict[str, tuple[str, str, bool]] = {
    "e2": ("contrastive margin m", r"m([\d.]+)$", False),
    "e3": ("triplet margin α", r"a([\d.]+)_", False),
    "e4": ("temperature τ", r"tau([\d.]+)$", True),
    "e5": ("negatives per anchor (batch)", r"batch(\d+)|global(\d+)", True),
    "e8": ("embedding dim d", r"dim(\d+)$", True),
}


def load_runs() -> pd.DataFrame:
    """Collect every completed run's metrics into one frame."""
    rows = []
    for metrics_path in sorted(RESULTS.glob("*/metrics.json")):
        name = metrics_path.parent.name
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        if "internal" not in data:
            continue

        config_path = metrics_path.parent / "config.yaml"
        cfg = {}
        if config_path.exists():
            import yaml

            cfg = yaml.safe_load(config_path.read_text())

        loss_cfg = cfg.get("loss", {})
        internal = data["internal"]
        lfw = data.get("lfw", {})
        retrieval = data.get("retrieval", {})
        representation = data.get("representation", {})

        rows.append(
            {
                "run": name,
                "experiment": name.split("_")[0],
                "loss": loss_cfg.get("name", ""),
                "miner": loss_cfg.get("miner", ""),
                "margin": loss_cfg.get("margin", ""),
                "tau": loss_cfg.get("temperature", ""),
                "negatives": loss_cfg.get("negatives", "local"),
                "dim": cfg.get("model", {}).get("embedding_dim", ""),
                "normalize": cfg.get("model", {}).get("normalize", ""),
                "aug": cfg.get("data", {}).get("augmentation", ""),
                "batch": data.get("global_batch_size", data.get("batch_size", "")),
                "lfw_mean": lfw.get("mean", float("nan")),
                "lfw_std": lfw.get("std", float("nan")),
                "auc": internal.get("auc", float("nan")),
                "tar@far1e-3": internal.get("tar@far=0.001", float("nan")),
                "tar@far1e-2": internal.get("tar@far=0.01", float("nan")),
                "eer": internal.get("eer", float("nan")),
                "accuracy": internal.get("best_accuracy", float("nan")),
                "recall@1": retrieval.get("recall@1", float("nan")),
                "recall@5": retrieval.get("recall@5", float("nan")),
                "alignment": representation.get("alignment", float("nan")),
                "uniformity": representation.get("uniformity", float("nan")),
                "epoch_s": data.get("median_epoch_seconds", float("nan")),
            }
        )

    return pd.DataFrame(rows).sort_values("run").reset_index(drop=True)


def write_master_table(frame: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "master_table.csv", index=False)

    columns = [
        "run", "loss", "lfw_mean", "lfw_std", "auc", "tar@far1e-3",
        "eer", "recall@1", "alignment", "uniformity",
    ]
    view = frame[columns].copy()
    for column in view.columns:
        if view[column].dtype.kind == "f":
            view[column] = view[column].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")

    lines = [
        "# Master comparison table",
        "",
        f"{len(frame)} runs. LFW is 10-fold mean ± std; every other column is the "
        "internal test-pair benchmark (6,000 pos + 6,000 neg, unseen identities).",
        "",
        "| " + " | ".join(view.columns) + " |",
        "|" + "|".join("---" for _ in view.columns) + "|",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    (OUT / "master_table.md").write_text("\n".join(lines) + "\n")


def _numeric(name: str, pattern: str) -> float | None:
    match = re.search(pattern, name)
    if not match:
        return None
    value = next((g for g in match.groups() if g), None)
    return float(value) if value is not None else None


def plot_sweeps(frame: pd.DataFrame) -> list[str]:
    """One V7 per experiment; E3 gets one line per miner."""
    written = []
    for experiment, (label, pattern, log_x) in SWEEPS.items():
        subset = frame[frame["experiment"] == experiment].copy()
        if subset.empty:
            continue
        subset["x"] = subset["run"].map(lambda n: _numeric(n, pattern))
        subset = subset.dropna(subset=["x"]).sort_values("x")
        if subset.empty:
            continue

        if experiment == "e3":
            series, x_values = {}, sorted(subset["x"].unique())
            for miner in ("random", "semi-hard", "batch-hard"):
                rows = subset[subset["miner"] == miner].sort_values("x")
                if len(rows) == len(x_values):
                    series[miner] = rows["lfw_mean"].tolist()
        else:
            x_values = subset["x"].tolist()
            series = {experiment.upper(): subset["lfw_mean"].tolist()}

        if not series:
            continue

        plot_sweep(
            x_values, series, OUT, name=f"v7_{experiment}",
            x_label=label, y_label="LFW accuracy (10-fold mean)",
            title=f"V7 · {experiment.upper()} — LFW vs {label}", log_x=log_x,
        )
        written.append(f"v7_{experiment}")
    return written


def plot_e1_roc(frame: pd.DataFrame) -> bool:
    """ROC overlay for the three E1 losses, recomputed from their checkpoints."""
    import torch

    from src.data.celeba import CelebAPairDataset
    from src.data.splits import load_eval_pairs
    from src.data.transforms import build_transform
    from src.engine.evaluate import embed_dataset, score_pairs
    from src.models.encoder import build_encoder
    from src.viz.roc import plot_roc

    runs = {
        "contrastive": RESULTS / "e1_contrastive" / "ckpts" / "best.pt",
        "triplet": RESULTS / "e1_triplet_semihard" / "ckpts" / "best.pt",
        "infonce": RESULTS / "e1_infonce" / "ckpts" / "best.pt",
    }
    if not all(p.exists() for p in runs.values()):
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs, labels = load_eval_pairs("data/splits", which="test")
    dataset = CelebAPairDataset(pairs, build_transform(train=False))

    curves = {}
    for name, ckpt in runs.items():
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        model = build_encoder(checkpoint["config"]["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        embeddings = embed_dataset(model, dataset, device, num_workers=8)
        curves[name] = (score_pairs(embeddings, dataset.pair_indices), labels)

    plot_roc(curves, OUT, name="v2_roc_e1", title="V2 · E1 loss face-off — ROC (log FPR)")
    return True


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    frame = load_runs()
    if frame.empty:
        print("no completed runs found")
        return

    write_master_table(frame)
    print(f"master table: {len(frame)} runs -> {OUT}/master_table.md")

    points = {
        row["run"]: (row["alignment"], row["uniformity"])
        for _, row in frame.iterrows()
        # E1 (the three losses) + E5 (the negatives sweep). E6 is excluded because
        # its two runs are the same experiment twice (see phase_5.md §d1) and their
        # coincident points make the labels unreadable.
        if pd.notna(row["alignment"]) and row["experiment"] in ("e1", "e5")
    }
    if points:
        plot_align_uniform(points, OUT, name="v6_align_uniform",
                           title="V6 · Alignment vs uniformity across losses")
        print(f"  v6_align_uniform ({len(points)} runs)")

    for name in plot_sweeps(frame):
        print(f"  {name}")

    if plot_e1_roc(frame):
        print("  v2_roc_e1")
    else:
        print("  v2_roc_e1 skipped (E1 checkpoints missing)")


if __name__ == "__main__":
    main()


_ = save  # re-exported for callers that save custom comparison figures
