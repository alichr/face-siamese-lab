"""Full evaluation + figure generation + auto `report.md` for one run (plan §8, §9).

Called at the end of every training job. Runs the complete measurement suite on
the best-by-val-AUC checkpoint and writes every number into `metrics.json` --
plan §14.5 requires that every figure in a report also exists as a number there,
so nothing in a report is unverifiable.

Evaluation order matters for honesty: the test pair list and LFW are touched
exactly once, here, after training and checkpoint selection are complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.celeba import CelebAIdentityDataset, CelebAPairDataset
from src.data.lfw import DEFAULT_LFW_ROOT, LFWImageDataset, load_lfw_pairs
from src.data.pk_sampler import PKSampler
from src.data.splits import load_eval_pairs, load_splits
from src.data.transforms import build_transform
from src.engine.evaluate import embed_dataset, score_pairs
from src.metrics.align_uniform import align_uniform
from src.metrics.retrieval import recall_at_k
from src.metrics.verification import lfw_10fold_accuracy, verification_metrics
from src.viz.curves import plot_loss_components, plot_training_curves
from src.viz.embedding_map import plot_embedding_map
from src.viz.heatmap import plot_similarity_heatmap
from src.viz.histograms import plot_similarity_histograms
from src.viz.roc import plot_roc
from src.viz.sweep_plots import plot_align_uniform

MAX_TEST_IDENTITIES = 400  # cap embedding cost for retrieval / diagnostics


def evaluate_run(
    model: torch.nn.Module,
    cfg: dict,
    out_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Run the full suite and write figures + `report.md`.

    Returns:
        The metrics dict (merged into `metrics.json` by the caller).
    """
    data_cfg = cfg["data"]
    celeba_root = Path(data_cfg["celeba_root"])
    figures = out_dir / "figures"
    workers = data_cfg["num_workers"]

    metrics: dict = {}
    eval_transform = build_transform(train=False)

    # --- internal verification on TEST identities (plan §8.1) ---------------
    pairs, labels = load_eval_pairs(data_cfg["splits_dir"], which="test")
    pair_set = CelebAPairDataset(pairs, eval_transform, root=celeba_root)
    pair_embeddings = embed_dataset(
        model, pair_set, device, num_workers=workers, amp_dtype=amp_dtype
    )
    scores = score_pairs(pair_embeddings, pair_set.pair_indices)
    internal = verification_metrics(scores, labels)
    metrics["internal"] = internal

    plot_similarity_histograms(
        scores,
        labels,
        figures,
        threshold=internal["best_threshold"],
        title="V1 · Internal test pairs — similarity distributions",
        subtitle=(
            f"AUC {internal['auc']:.4f} · EER {internal['eer']:.4f} · "
            f"TAR@FAR=1e-3 {internal['tar@far=0.001']:.4f}"
        ),
    )

    roc_curves = {"internal (CelebA test ids)": (scores, labels)}

    # --- LFW 10-fold (plan §8.2) --------------------------------------------
    try:
        lfw = load_lfw_pairs(DEFAULT_LFW_ROOT)
        lfw_set = LFWImageDataset(lfw, eval_transform, root=DEFAULT_LFW_ROOT)
        lfw_embeddings = embed_dataset(
            model, lfw_set, device, num_workers=workers, amp_dtype=amp_dtype
        )
        lfw_scores = score_pairs(lfw_embeddings, lfw_set.pair_indices)
        metrics["lfw"] = lfw_10fold_accuracy(lfw_scores, lfw.labels, lfw.folds)
        metrics["lfw"].update(
            {f"verification_{k}": v for k, v in verification_metrics(lfw_scores, lfw.labels).items()}
        )
        roc_curves["LFW"] = (lfw_scores, lfw.labels)
    except FileNotFoundError as exc:
        metrics["lfw"] = {"error": str(exc)}

    plot_roc(roc_curves, figures, title="V2 · ROC (log FPR)")

    # --- test-identity embeddings for retrieval / diagnostics / UMAP --------
    splits = load_splits(data_cfg["splits_dir"])
    test_set = CelebAIdentityDataset(
        identities=splits.test,
        transform=eval_transform,
        root=celeba_root,
        max_identities=MAX_TEST_IDENTITIES,
    )

    class _Indexed(torch.utils.data.Dataset):
        """Adapt (image, label) -> (image, row) for `embed_dataset`."""

        def __len__(self) -> int:
            return len(test_set)

        def __getitem__(self, i):
            image, _ = test_set[i]
            return image, i

    test_embeddings = embed_dataset(
        model, _Indexed(), device, num_workers=workers, amp_dtype=amp_dtype
    )
    test_labels = test_set.labels

    metrics["retrieval"] = recall_at_k(test_embeddings, test_labels)
    metrics["representation"] = align_uniform(test_embeddings, test_labels)

    plot_embedding_map(
        test_embeddings, test_labels, figures, title="V4 · UMAP — 30 unseen identities"
    )
    plot_align_uniform(
        {cfg["loss"]["name"]: (
            metrics["representation"]["alignment"],
            metrics["representation"]["uniformity"],
        )},
        figures,
        title="V6 · Alignment vs uniformity",
    )

    # --- V3: similarity heatmap on a real PK batch --------------------------
    heatmap_sampler = PKSampler(test_labels, p=16, k=4, seed=0, num_batches=1)
    batch_rows = next(iter(heatmap_sampler))
    plot_similarity_heatmap(
        test_embeddings[batch_rows],
        test_labels[batch_rows],
        figures,
        title="V3 · Batch similarity matrix S (identity-ordered)",
    )

    # --- V5: training curves -------------------------------------------------
    curves_csv = out_dir / "curves.csv"
    if curves_csv.exists():
        plot_training_curves(curves_csv, figures, title="V5 · Training curves")
        plot_loss_components(curves_csv, figures)

    write_report(out_dir, cfg, metrics)
    return metrics


def write_report(out_dir: Path, cfg: dict, metrics: dict) -> Path:
    """Render `report.md` for a single run (plan §7)."""
    loss_cfg = cfg["loss"]
    internal = metrics.get("internal", {})
    lfw = metrics.get("lfw", {})
    retrieval = metrics.get("retrieval", {})
    representation = metrics.get("representation", {})

    descriptor = loss_cfg["name"]
    if loss_cfg["name"] == "triplet":
        descriptor += f" (miner={loss_cfg.get('miner')}, α={loss_cfg.get('margin')})"
    elif loss_cfg["name"] == "contrastive":
        descriptor += f" (m={loss_cfg.get('margin')})"
    else:
        descriptor += f" (τ={loss_cfg.get('temperature')})"

    lines = [
        f"# {Path(out_dir).name}",
        "",
        f"**Loss:** {descriptor}  ",
        f"**Model:** {cfg['model']['backbone']}, d={cfg['model']['embedding_dim']}, "
        f"normalize={cfg['model']['normalize']}  ",
        f"**Sampler:** P={cfg['sampler']['p']} × K={cfg['sampler']['k']} "
        f"= {cfg['sampler']['p'] * cfg['sampler']['k']}  ",
        f"**Schedule:** {cfg['train']['epochs']} epochs, lr {cfg['train']['lr']}, "
        f"aug={cfg['data']['augmentation']}, amp={cfg['train']['amp']}",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---|",
    ]

    if isinstance(lfw, dict) and "mean" in lfw:
        lines.append(f"| **LFW accuracy (10-fold)** | **{lfw['mean']:.4f} ± {lfw['std']:.4f}** |")
    if internal:
        lines += [
            f"| internal ROC-AUC | {internal['auc']:.4f} |",
            f"| internal TAR@FAR=1e-3 | {internal['tar@far=0.001']:.4f} |",
            f"| internal TAR@FAR=1e-2 | {internal['tar@far=0.01']:.4f} |",
            f"| internal EER | {internal['eer']:.4f} |",
            f"| internal best-threshold accuracy | {internal['best_accuracy']:.4f} |",
            f"| mean s_pos − s_neg | {internal['gap']:.4f} |",
        ]
    for k in ("recall@1", "recall@5", "recall@10"):
        if k in retrieval:
            lines.append(f"| {k} | {retrieval[k]:.4f} |")
    if representation:
        lines += [
            f"| alignment (lower better) | {representation['alignment']:.4f} |",
            f"| uniformity (lower better) | {representation['uniformity']:.4f} |",
        ]

    if isinstance(lfw, dict) and "folds" in lfw:
        lines += [
            "",
            "## LFW 10-fold detail",
            "",
            "Threshold fit on 9 folds, tested on the held-out 10th (plan §8.2).",
            "",
            "| fold | accuracy |",
            "|---|---|",
        ]
        lines += [f"| {i} | {a:.4f} |" for i, a in enumerate(lfw["folds"])]
        lines.append(f"| **mean ± std** | **{lfw['mean']:.4f} ± {lfw['std']:.4f}** |")

    lines += [
        "",
        "## Figures",
        "",
        "| ID | Figure | File |",
        "|---|---|---|",
        "| V1 | pos/neg similarity histograms + threshold | `figures/v1_similarity_histograms.png` |",
        "| V2 | ROC, log FPR | `figures/v2_roc.png` |",
        "| V3 | batch similarity matrix S | `figures/v3_similarity_heatmap.png` |",
        "| V4 | UMAP, 30 unseen identities | `figures/v4_umap.png` |",
        "| V5 | training curves | `figures/v5_training_curves.png` |",
        "| V6 | alignment–uniformity | `figures/v6_align_uniform.png` |",
        "",
        "All figures also exist as PDF. Every number above is in `metrics.json`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f".venv/bin/python -m src.engine.train --config {out_dir}/config.yaml",
        "```",
    ]

    path = Path(out_dir) / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
