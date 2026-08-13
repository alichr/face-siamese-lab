"""Single-GPU training loop (plan §7, poster panel 5).

    sample PK batch -> encode with shared f_theta -> loss -> backprop -> repeat

Everything is YAML-driven and the *resolved* config (defaults merged with the
file) is written next to the outputs, so a results folder can be replayed
without consulting the code that produced it (plan §14.3).

Phase 1 wires only the contrastive loss; triplet and InfoNCE join in Phase 2 via
`build_loss`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.celeba import CelebAIdentityDataset
from src.data.pk_sampler import PKSampler
from src.data.splits import load_eval_pairs, load_splits
from src.data.transforms import build_transform
from src.engine.evaluate import evaluate_pairs
from src.engine.report import evaluate_run
from src.losses.contrastive import ContrastiveLoss
from src.losses.infonce import InfoNCELoss
from src.losses.triplet import TripletLoss
from src.models.encoder import build_encoder

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "output_dir": None,
    "data": {
        "celeba_root": "data/celeba",
        "splits_dir": "data/splits",
        "augmentation": "basic",
        "num_workers": 8,
        "debug_identities": None,  # E0 uses 10
        "eval_split": "val",
    },
    "model": {
        "backbone": "resnet18",
        "embedding_dim": 128,
        "normalize": True,
        "pretrained": False,
    },
    "loss": {"name": "contrastive", "margin": 1.0},
    "sampler": {"p": 64, "k": 4, "allow_replacement": False},
    "train": {
        "epochs": 30,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "grad_clip": 5.0,
        "amp": "bf16",
        "eval_every": 1,
        "batches_per_epoch": None,
        "full_eval": True,
    },
}

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None, None: None}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path, overrides: dict | None = None) -> dict:
    """Load a YAML config, merged onto DEFAULTS (plan §14.3: no hyperparameter lives only in code)."""
    with Path(path).open() as f:
        cfg = yaml.safe_load(f) or {}
    resolved = deep_merge(DEFAULTS, cfg)
    if overrides:
        resolved = deep_merge(resolved, overrides)
    if not resolved.get("output_dir"):
        resolved["output_dir"] = f"results/{Path(path).stem}"
    return resolved


def seed_everything(seed: int) -> None:
    """Seed torch, numpy and random (plan §2)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(cfg: dict):
    """Construct the loss named in the config's `loss` block (plan §6)."""
    name = cfg["name"]
    seed = cfg.get("seed", 0)

    if name == "contrastive":
        return ContrastiveLoss(margin=cfg.get("margin", 1.0), seed=seed)
    if name == "triplet":
        return TripletLoss(
            margin=cfg.get("margin", 0.2), miner=cfg.get("miner", "semi-hard"), seed=seed
        )
    if name == "infonce":
        return InfoNCELoss(
            temperature=cfg.get("temperature", 0.07),
            supcon=cfg.get("supcon", False),
            seed=seed,
        )
    raise ValueError(f"unknown loss {name!r}; expected one of contrastive/triplet/infonce")


def build_lr_lambda(warmup_epochs: int, total_epochs: int, batches_per_epoch: int):
    """Linear warmup over `warmup_epochs`, then cosine decay to zero (plan §7)."""
    warmup_steps = max(1, warmup_epochs * batches_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * batches_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    return lr_lambda


def train(config_path: Path, overrides: dict | None = None) -> dict:
    """Run one training job end to end.

    Returns:
        The final metrics dict (also written to `metrics.json`).
    """
    cfg = load_config(config_path, overrides)
    seed_everything(cfg["seed"])

    out_dir = Path(cfg["output_dir"])
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # --- data ---------------------------------------------------------------
    data_cfg = cfg["data"]
    splits = load_splits(data_cfg["splits_dir"])  # asserts disjointness (plan §14.4)
    print(splits.summary(), flush=True)

    train_set = CelebAIdentityDataset(
        identities=splits.train,
        transform=build_transform(data_cfg["augmentation"], train=True),
        root=Path(data_cfg["celeba_root"]),
        max_identities=data_cfg["debug_identities"],
    )
    print(f"train: {train_set.summary()}", flush=True)

    sampler_cfg = cfg["sampler"]

    # An "epoch" means one pass over the training IMAGES, not over identities.
    # With P=64, K=4 a single pass over identities is only 7,512/64 = 117 batches
    # = 29,952 images -- under 20% of the split -- so 30 such epochs would be
    # ~5.7 dataset passes and would badly undertrain relative to the plan's
    # 0.85-0.94 LFW expectation (and its ~1-2 h/run estimate, which only makes
    # sense at full-split epochs). The sampler reshuffles identities whenever it
    # exhausts them, so asking for more batches than P-groups is well-defined.
    batches_per_epoch = cfg["train"]["batches_per_epoch"] or max(
        1, len(train_set) // (sampler_cfg["p"] * sampler_cfg["k"])
    )

    sampler = PKSampler(
        train_set.labels,
        p=sampler_cfg["p"],
        k=sampler_cfg["k"],
        seed=cfg["seed"],
        allow_replacement=sampler_cfg["allow_replacement"],
        num_batches=batches_per_epoch,
    )
    print(
        f"sampler: P={sampler.p} K={sampler.k} batch={sampler.batch_size} "
        f"{len(sampler)} batches/epoch, {sampler.n_dropped_identities} ids dropped (<K images)",
        flush=True,
    )

    loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=data_cfg["num_workers"] > 0,
    )

    # E0 overfits the debug identities, so it must be *scored* on them too --
    # measuring generalization would defeat the purpose of a sanity check.
    if data_cfg["debug_identities"]:
        eval_pairs, eval_labels = _debug_pairs(train_set, seed=cfg["seed"])
        eval_tag = f"debug({data_cfg['debug_identities']} ids)"
    else:
        eval_pairs, eval_labels = load_eval_pairs(
            data_cfg["splits_dir"], which=data_cfg["eval_split"]
        )
        eval_tag = data_cfg["eval_split"]
    print(f"eval on {eval_tag}: {len(eval_pairs):,} pairs", flush=True)

    # --- model, loss, optimizer ---------------------------------------------
    model = build_encoder(cfg["model"]).to(device)
    print(model.summary(), flush=True)

    criterion = build_loss({**cfg["loss"], "seed": cfg["seed"]})

    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(train_cfg["warmup_epochs"], train_cfg["epochs"], len(sampler)),
    )
    amp_dtype = AMP_DTYPES[train_cfg["amp"]]

    # --- loop ---------------------------------------------------------------
    curves_path = out_dir / "curves.csv"
    curves_file = curves_path.open("w", newline="")
    writer: csv.DictWriter | None = None

    best_auc, best_epoch = -1.0, -1
    epoch_times: list[float] = []
    last_eval: dict[str, float] = {}

    for epoch in range(train_cfg["epochs"]):
        sampler.set_epoch(epoch)
        model.train()

        epoch_start = time.perf_counter()
        running: dict[str, float] = {}
        n_steps = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    embeddings = model(images)
                # Losses run in fp32: the similarity matrix and the log-sum-exp
                # in InfoNCE lose meaningful precision in bf16 (~3 decimal digits).
                loss, logs = criterion(embeddings.float(), labels)
            else:
                embeddings = model(images)
                loss, logs = criterion(embeddings, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            running["loss"] = running.get("loss", 0.0) + loss.item()
            for key, value in logs.items():
                running[key] = running.get(key, 0.0) + value
            n_steps += 1

        torch.cuda.synchronize() if device.type == "cuda" else None
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "epoch_seconds": epoch_time}
        row.update({k: v / max(1, n_steps) for k, v in running.items()})
        row["img_per_s"] = len(sampler) * sampler.batch_size / epoch_time

        if (epoch + 1) % train_cfg["eval_every"] == 0 or epoch == train_cfg["epochs"] - 1:
            last_eval = evaluate_pairs(
                model,
                eval_pairs,
                eval_labels,
                device,
                celeba_root=Path(data_cfg["celeba_root"]),
                num_workers=data_cfg["num_workers"],
                amp_dtype=amp_dtype,
            )
            row.update({f"val_{k}": v for k, v in last_eval.items()})

            if last_eval["auc"] > best_auc:
                best_auc, best_epoch = last_eval["auc"], epoch
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "auc": best_auc, "config": cfg},
                    out_dir / "ckpts" / "best.pt",
                )

        if writer is None:
            writer = csv.DictWriter(curves_file, fieldnames=list(row))
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        curves_file.flush()

        # Print every diagnostic the loss reported. `active_fraction` in
        # particular is the curve that separates the three miners in E3, so it
        # must never be filtered out of the console log.
        diagnostics = " ".join(
            f"{k} {row[k]:.4f}"
            for k in row
            if k not in ("epoch", "lr", "loss", "epoch_seconds", "img_per_s")
            and not k.startswith("val_")
        )
        print(
            f"epoch {epoch:3d} | loss {row['loss']:.4f} | {diagnostics}"
            + (f" | val_auc {row['val_auc']:.4f}" if "val_auc" in row else "")
            + f" | {epoch_time:.1f}s {row['img_per_s']:.0f} img/s",
            flush=True,
        )

    curves_file.close()
    torch.save(
        {"model": model.state_dict(), "epoch": train_cfg["epochs"] - 1, "config": cfg},
        out_dir / "ckpts" / "last.pt",
    )

    # Full evaluation runs on the BEST-by-val-AUC checkpoint, not the last one.
    # The test pair list and LFW are touched exactly once, here, after training
    # and checkpoint selection are both finished.
    best_path = out_dir / "ckpts" / "best.pt"
    if best_path.exists() and train_cfg["full_eval"]:
        model.load_state_dict(torch.load(best_path, weights_only=False)["model"])

    full_metrics = (
        evaluate_run(model, cfg, out_dir, device, amp_dtype=amp_dtype)
        if train_cfg["full_eval"]
        else {}
    )

    metrics = {
        **full_metrics,
        "final": last_eval,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "final_train_loss": row["loss"],
        "epochs": train_cfg["epochs"],
        "median_epoch_seconds": float(np.median(epoch_times)),
        "median_img_per_s": float(
            len(sampler) * sampler.batch_size / np.median(epoch_times)
        ),
        "batch_size": sampler.batch_size,
        "batches_per_epoch": len(sampler),
        "train_images": len(train_set),
        "train_identities": train_set.n_identities,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def _debug_pairs(dataset: CelebAIdentityDataset, seed: int, n_per_class: int = 500):
    """Build verification pairs from the debug identities themselves (E0 only)."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[str]] = {}
    for filename, label in zip(dataset.files, dataset.labels):
        by_label.setdefault(int(label), []).append(filename)

    eligible = [lbl for lbl, files in by_label.items() if len(files) >= 2]
    positives, negatives = set(), set()

    while len(positives) < n_per_class:
        label = eligible[rng.integers(len(eligible))]
        files = by_label[label]
        a, b = rng.choice(len(files), size=2, replace=False)
        positives.add(tuple(sorted((files[a], files[b]))))

    labels_list = sorted(by_label)
    while len(negatives) < n_per_class:
        i, j = rng.choice(len(labels_list), size=2, replace=False)
        a = by_label[labels_list[i]][rng.integers(len(by_label[labels_list[i]]))]
        b = by_label[labels_list[j]][rng.integers(len(by_label[labels_list[j]]))]
        negatives.add(tuple(sorted((a, b))))

    pairs = [*sorted(positives), *sorted(negatives)]
    labels = [1] * len(positives) + [0] * len(negatives)
    return pairs, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    overrides: dict[str, Any] = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["train"] = {"epochs": args.epochs}

    train(args.config, overrides)


if __name__ == "__main__":
    main()
