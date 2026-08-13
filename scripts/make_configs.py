"""Emit every E1-E8 config from the plan §10 matrix (committed to `configs/`).

Generated rather than hand-written so that the *only* thing differing between
runs is the one variable each experiment sweeps. Hand-editing 35 YAML files is
how a stray `epochs: 20` ends up in one row and quietly invalidates its column.

    .venv/bin/python scripts/make_configs.py

E5's global-negatives row is the single `torchrun` job; everything else is
single-GPU (plan §2).
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path("configs")

BASE: dict = {
    "seed": 0,
    "data": {
        "celeba_root": "data/celeba",
        "splits_dir": "data/splits",
        "augmentation": "basic",
        "num_workers": 8,
        "debug_identities": None,
        "eval_split": "val",
    },
    "model": {
        "backbone": "resnet18",
        "embedding_dim": 128,
        "normalize": True,
        "pretrained": False,
    },
    "loss": {"name": "infonce", "temperature": 0.07, "supcon": False, "negatives": "local"},
    "sampler": {"p": 64, "k": 4, "allow_replacement": False},
    "train": {
        "epochs": 30,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 5,
        "grad_clip": 5.0,
        "amp": "bf16",
        "eval_every": 1,
        "batches_per_epoch": None,
        "full_eval": True,
        "sync_bn": True,
    },
}


def merge(base: dict, override: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def build() -> dict[str, dict]:
    """Return {config_name: config}. Names encode the experiment and its variable."""
    configs: dict[str, dict] = {}

    # --- E1: loss face-off at identical budget -----------------------------
    configs["e1_contrastive"] = merge(BASE, {"loss": {"name": "contrastive", "margin": 1.0}})
    configs["e1_triplet_semihard"] = merge(
        BASE, {"loss": {"name": "triplet", "margin": 0.2, "miner": "semi-hard"}}
    )
    configs["e1_infonce"] = merge(BASE, {})

    # --- E2: contrastive margin (m <= 2 on the unit sphere) ----------------
    for m in (0.25, 0.5, 1.0, 1.5):
        configs[f"e2_contrastive_m{m}"] = merge(
            BASE, {"loss": {"name": "contrastive", "margin": m}}
        )

    # --- E3: triplet margin x miner ----------------------------------------
    for alpha in (0.1, 0.2, 0.4):
        for miner in ("random", "semi-hard", "batch-hard"):
            configs[f"e3_triplet_a{alpha}_{miner}"] = merge(
                BASE, {"loss": {"name": "triplet", "margin": alpha, "miner": miner}}
            )

    # --- E4: InfoNCE temperature -------------------------------------------
    for tau in (0.03, 0.05, 0.07, 0.1, 0.2, 0.5):
        configs[f"e4_infonce_tau{tau}"] = merge(BASE, {"loss": {"temperature": tau}})

    # --- E5: negatives / batch size ----------------------------------------
    # P is halved/quartered rather than K, so every batch keeps K=4 and the
    # positive structure is identical -- only the NUMBER of negatives changes.
    for p, batch in ((16, 64), (32, 128), (64, 256)):
        configs[f"e5_infonce_batch{batch}"] = merge(BASE, {"sampler": {"p": p}})
    configs["e5_infonce_global768"] = merge(BASE, {"loss": {"negatives": "global"}})

    # --- E6: L2 normalization on/off ---------------------------------------
    configs["e6_norm_on"] = merge(BASE, {"model": {"normalize": True}})
    configs["e6_norm_off"] = merge(BASE, {"model": {"normalize": False}})

    # --- E7: augmentation ---------------------------------------------------
    for level in ("none", "basic", "strong"):
        configs[f"e7_aug_{level}"] = merge(BASE, {"data": {"augmentation": level}})

    # --- E8: embedding dimension -------------------------------------------
    for d in (64, 128, 256, 512):
        configs[f"e8_dim{d}"] = merge(BASE, {"model": {"embedding_dim": d}})

    for name, cfg in configs.items():
        cfg["output_dir"] = f"results/{name}"

    return configs


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    configs = build()
    for name, cfg in sorted(configs.items()):
        (OUT_DIR / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    ddp = [n for n, c in configs.items() if c["loss"].get("negatives") == "global"]
    print(f"wrote {len(configs)} configs to {OUT_DIR}/")
    print(f"  single-GPU : {len(configs) - len(ddp)}")
    print(f"  torchrun   : {len(ddp)}  {ddp}")


if __name__ == "__main__":
    main()
