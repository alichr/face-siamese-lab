"""Poster panel 6, made tangible: verify two face images (plan §8.5).

    .venv/bin/python scripts/verify_pair.py A.jpg B.jpg --ckpt results/<run>/ckpts/best.pt

Prints the cosine similarity, the threshold, and **Accept** or **Reject**, and
saves a side-by-side figure -- exactly the panel-6 layout.

The threshold is not invented here. It comes from the run's `metrics.json`,
where it was fit on the internal test pairs; passing `--threshold` overrides it.
A verification system without a stated, separately-fitted threshold is not a
system, it is a similarity function -- the choice of operating point IS the
product decision, which is why panel 6 draws t explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import build_transform
from src.losses.geometry import cosine_similarity
from src.models.encoder import build_encoder
from src.viz.style import CATEGORICAL, TEXT_PRIMARY, apply_style

ACCEPT_COLOR = "#008300"
REJECT_COLOR = "#e34948"


def load_model(ckpt_path: Path, device: torch.device):
    """Load an encoder from a checkpoint, returning `(model, config)`."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = build_encoder(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg


def resolve_threshold(ckpt_path: Path, override: float | None) -> tuple[float, str]:
    """Return `(threshold, provenance)`, preferring the run's fitted value."""
    if override is not None:
        return override, "command line"

    metrics_path = Path(ckpt_path).parent.parent / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        internal = metrics.get("internal", {})
        if "best_threshold" in internal:
            return float(internal["best_threshold"]), f"fitted on internal test pairs ({metrics_path})"

    return 0.5, "default fallback (no metrics.json found)"


@torch.no_grad()
def verify(image_a: Path, image_b: Path, ckpt: Path, threshold: float | None, out: Path | None):
    """Score one pair and report the decision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt, device)
    transform = build_transform(train=False)

    pil_a = Image.open(image_a).convert("RGB")
    pil_b = Image.open(image_b).convert("RGB")
    batch = torch.stack([transform(pil_a), transform(pil_b)]).to(device)

    embeddings = model(batch).float().cpu()
    similarity = float(cosine_similarity(embeddings[0:1], embeddings[1:2]).item())
    distance = float((2.0 - 2.0 * similarity) ** 0.5)

    t, provenance = resolve_threshold(ckpt, threshold)
    accept = similarity > t

    print()
    print(f"  image A     : {image_a}")
    print(f"  image B     : {image_b}")
    print(f"  checkpoint  : {ckpt}")
    print(f"  similarity  : s = {similarity:.4f}   (equivalently d = {distance:.4f})")
    print(f"  threshold   : t = {t:.4f}   [{provenance}]")
    print(f"  decision    : {'ACCEPT (same person)' if accept else 'REJECT (different person)'}")
    print()

    if out is not None:
        _save_figure(pil_a, pil_b, similarity, t, accept, out)
        print(f"  figure      : {out}\n")

    return {"similarity": similarity, "distance": distance, "threshold": t, "accept": accept}


def _save_figure(pil_a, pil_b, similarity: float, threshold: float, accept: bool, out: Path) -> None:
    """Side-by-side figure in the poster-panel-6 layout."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.0))
    for ax, image, label in zip(axes, (pil_a, pil_b), ("A", "B")):
        ax.imshow(image)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)

    color = ACCEPT_COLOR if accept else REJECT_COLOR
    verdict = "ACCEPT — same person" if accept else "REJECT — different person"
    comparator = ">" if accept else "<"

    fig.suptitle(
        f"s = {similarity:.3f}  {comparator}  t = {threshold:.3f}",
        fontsize=11,
        fontweight="semibold",
        color=TEXT_PRIMARY,
        y=1.06,
    )
    fig.text(0.5, 0.94, verdict, ha="center", fontsize=10, color=color, fontweight="bold")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("img1", type=Path)
    ap.add_argument("img2", type=Path)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/verify_pair.png"))
    args = ap.parse_args()

    verify(args.img1, args.img2, args.ckpt, args.threshold, args.out)


if __name__ == "__main__":
    main()


# Silence the unused-import warning for the palette module, which is imported
# for its side effect of registering the shared style.
_ = CATEGORICAL
