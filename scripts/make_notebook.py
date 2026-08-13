"""Generate the interactive tour notebook (`notebooks/siamese_lab_tour.ipynb`).

    .venv/bin/python scripts/make_notebook.py

The notebook is generated rather than hand-edited so it can be regenerated when
the API moves, and so it stays reviewable as source in git (a committed .ipynb
diffs terribly). Edit this file, re-run it, and the notebook follows.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/siamese_lab_tour.ipynb")

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


# =============================================================================
md(r"""
# Training a Siamese Network — an interactive tour

This notebook walks the **entire** lab, from raw pixels to the final findings, in the
order the ideas actually depend on each other:

| § | Topic | The question it answers |
|---|---|---|
| 0 | Setup | Is everything installed and on disk? |
| 1 | Data | How are identity splits, PK batches and augmentation built — and why? |
| 2 | Geometry | Why are "cosine similarity" and "Euclidean distance" the same thing? |
| 3 | The encoder | What does "shared weights" actually mean in code? |
| 4 | Losses | Contrastive vs triplet vs InfoNCE, with their exact formulas |
| 5 | Mining | Why the miner matters more than the margin |
| 6 | Training | Watch a model learn, live |
| 7 | Evaluation | Threshold, ROC, TAR@FAR, EER, LFW 10-fold, Recall@K |
| 8 | Diagnostics | Alignment & uniformity: *why* InfoNCE behaves as it does |
| 9 | Distributed | The silent bug that cross-GPU negatives can hide |
| 10 | Results | Explore every experiment in the matrix |

**Every cell is meant to be edited.** Cells marked 🎛 **KNOB** have a value at the top
worth changing — the point is to break things and see what happens.

Reference documents: [`docs/plan.md`](../docs/plan.md) (the spec),
[`docs/poster.png`](../docs/poster.png) (the source poster),
[`results/phase_reports/`](../results/phase_reports/) (what was found at each stage).
""")

# =============================================================================
md(r"""
---
## 0 · Setup

Run this first. It moves the working directory to the repo root so that every
relative path (`data/…`, `results/…`) resolves the way the library expects.
""")

code(r"""
import os, sys
from pathlib import Path

# Walk up to the repo root (the directory holding pyproject.toml).
here = Path.cwd()
while not (here / "pyproject.toml").exists() and here != here.parent:
    here = here.parent
os.chdir(here)
sys.path.insert(0, str(here))
print("repo root:", here)

import numpy as np
import torch

# Import the shared figure style FIRST, then re-enable the inline backend.
# `src.viz.style` calls matplotlib.use("Agg") so training runs work headless on
# a server -- which would otherwise silently swallow every plot in this
# notebook. Order matters: the magic below overrides Agg.
from src.viz.style import apply_style
apply_style()

get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
print("matplotlib backend:", plt.get_backend())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} · device {DEVICE} · {torch.cuda.device_count()} GPU(s)")
if torch.cuda.is_available():
    print(" ", torch.cuda.get_device_name(0))
""")

code(r"""
# What is on disk? Nothing below will work without these.
for label, path in [
    ("CelebA images",  "data/celeba/img_align_celeba"),
    ("CelebA identities", "data/celeba/identity_CelebA.txt"),
    ("LFW (funneled)", "data/lfw_sklearn/lfw_home/lfw_funneled"),
    ("splits",         "data/splits/train_identities.txt"),
    ("baseline run",   "results/baseline_infonce/ckpts/best.pt"),
]:
    print(f"{'OK ' if Path(path).exists() else 'MISSING'}  {label:20s} {path}")
""")

# =============================================================================
md(r"""
---
## 1 · Data

### 1.1 Identity-disjoint splits — the most important design decision

The splits are by **identity**, never by image. This is the single choice that makes
every number in this lab meaningful.

Suppose you split by *image* instead: the same person appears in both train and test.
Verification then asks "are these two faces the same person?" about identities the
encoder was explicitly trained to place at a known spot on the sphere. It can answer
from memorised identity rather than from any transferable notion of facial similarity.
The reported accuracy would be a measure of recall on training identities, and it would
collapse on a face the system had never seen.

Because that failure makes results look **better**, it gets an assertion, not a comment.
""")

code(r"""
from src.data.splits import load_splits, load_eval_pairs

splits = load_splits()
print(splits.summary())
print()
for name in ("train", "val", "test"):
    ids = getattr(splits, name)
    print(f"  {name:5s} {len(ids):>6,} identities   e.g. {sorted(ids)[:5]}")
""")

code(r"""
# 🎛 KNOB — try to sneak a training identity into the test set and watch it fail loudly.
from src.data.splits import IdentitySplits, assert_identity_disjoint

leaky = IdentitySplits(
    train=frozenset({1, 2, 3}),
    val=frozenset({4, 5}),
    test=frozenset({3, 6}),      # <-- identity 3 is also in train
)
try:
    assert_identity_disjoint(leaky)
    print("no leak detected (this line should never print)")
except ValueError as exc:
    print("caught:\n ", exc)
""")

md(r"""
### 1.2 The two evaluation pair lists

There are **two** fixed pair lists, and confusing them would quietly bias every result.

* `val` (3,000 + 3,000 pairs, *val* identities) — scored every epoch, used to pick the
  best checkpoint.
* `test` (6,000 + 6,000 pairs, *test* identities) — the final benchmark, touched **once**
  per run, after training and checkpoint selection are both finished.

If you selected the checkpoint on the test list, the reported number would include the
gain from picking whichever epoch happened to suit those exact pairs.
""")

code(r"""
val_pairs, val_labels = load_eval_pairs(which="val")
test_pairs, test_labels = load_eval_pairs(which="test")

print(f"val  : {len(val_pairs):,} pairs  ({sum(val_labels):,} positive)")
print(f"test : {len(test_pairs):,} pairs  ({sum(test_labels):,} positive)")

val_files = {f for p in val_pairs for f in p}
test_files = {f for p in test_pairs for f in p}
print(f"\nimages shared between the two lists: {len(val_files & test_files)}  (must be 0)")
print("example positive pair:", test_pairs[0])
print("example negative pair:", test_pairs[-1])
""")

md(r"""
### 1.3 Look at the actual faces

Abstractions are easier to trust once you have seen the inputs.
""")

code(r"""
from PIL import Image
from src.data.celeba import files_by_identity

by_id = files_by_identity()
some_ids = sorted(splits.test)[:4]

fig, axes = plt.subplots(len(some_ids), 6, figsize=(9, 1.6 * len(some_ids)))
for row, identity in enumerate(some_ids):
    for col in range(6):
        ax = axes[row, col]
        files = by_id[identity]
        if col < len(files):
            ax.imshow(Image.open(f"data/celeba/img_align_celeba/{files[col]}"))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if col == 0:
            ax.set_ylabel(f"id {identity}", fontsize=8)
fig.suptitle("Same row = same identity (pose, lighting, expression vary)", y=1.01)
plt.tight_layout(); plt.show()
""")

md(r"""
### 1.4 Augmentation levels 🎛

`none` is **crop only** — note that it still random-crops, which is why the E0 overfit
run bottoms out near loss 0.06 rather than exactly 0.

Change `LEVEL` and re-run to see what the network is asked to be invariant to.
""")

code(r"""
LEVEL = "strong"        # 🎛 KNOB: "none" | "basic" | "strong"

from src.data.transforms import build_transform, AUGMENTATION_LEVELS
print("available:", AUGMENTATION_LEVELS)

transform = build_transform(LEVEL, train=True)
image = Image.open(f"data/celeba/img_align_celeba/{by_id[some_ids[0]][0]}")

fig, axes = plt.subplots(1, 8, figsize=(12, 1.8))
for ax in axes:
    tensor = transform(image)                 # random every call
    ax.imshow((tensor.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy())
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle(f"Eight draws of the SAME image, augmentation = '{LEVEL}'", y=1.08)
plt.show()
""")

md(r"""
### 1.5 PK sampling — why every batch is P identities × K images

All three losses need *positives inside the batch*:

* **contrastive** needs same-identity pairs to pull together,
* **triplet** needs an anchor's positive to exist before "hardest positive" means anything,
* **InfoNCE** needs one in-batch positive per anchor.

With `K = 1` a batch would be P singleton identities and there would be **no positive
pairs at all**. The sampler is what makes the in-batch positive exist.
""")

code(r"""
P, K = 8, 4              # 🎛 KNOB: try K=2, or P=4. K=1 raises — read the error.

from src.data.pk_sampler import PKSampler
from src.data.celeba import CelebAIdentityDataset

train_set = CelebAIdentityDataset(
    identities=splits.train, transform=build_transform("basic", train=True), max_identities=60
)
sampler = PKSampler(train_set.labels, p=P, k=K, seed=0)

print(f"{sampler.n_identities} usable identities, {sampler.n_dropped_identities} dropped (<K images)")
print(f"batch size = P*K = {sampler.batch_size}, {len(sampler)} batches/epoch\n")

batch = next(iter(sampler))
ids, counts = np.unique(train_set.labels[np.array(batch)], return_counts=True)
print(f"one batch: {len(batch)} images, {len(ids)} identities, per-identity counts {set(counts)}")
""")

code(r"""
# See the batch. Each row is one identity's K images -- the block structure that
# makes the similarity matrix in §4.3 block-diagonal.
rows = np.array(batch).reshape(P, K)
fig, axes = plt.subplots(P, K, figsize=(1.3 * K, 1.3 * P))
for r in range(P):
    for c in range(K):
        img, label = train_set[rows[r, c]]
        axes[r, c].imshow((img.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy())
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
    axes[r, 0].set_ylabel(f"id {label}", fontsize=7)
fig.suptitle(f"One PK batch: {P} identities x {K} images", y=1.01)
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""
---
## 2 · Geometry — the identity everything rests on

For L2-normalised embeddings:

$$d(z_1,z_2)^2 = \|z_1-z_2\|^2 = \|z_1\|^2 + \|z_2\|^2 - 2 z_1^\top z_2 = 2 - 2s(z_1,z_2)$$

So $d$ is a strictly decreasing function of $s$. **Thresholding one is thresholding the
other** — which is why the poster's "cosine similarity" box and "Euclidean distance" box
are two views of one quantity. Contrastive and triplet are written in $d$; InfoNCE in
$s$; on the unit sphere they rank the same pairs.
""")

code(r"""
from src.losses.geometry import (
    l2_normalize, cosine_similarity, euclidean_distance,
    cosine_similarity_matrix, euclidean_distance_matrix, distance_from_similarity,
)

z1 = l2_normalize(torch.randn(1000, 128))
z2 = l2_normalize(torch.randn(1000, 128))
s = cosine_similarity(z1, z2)
d = euclidean_distance(z1, z2)

print(f"max |d^2 - (2 - 2s)| = {(d.pow(2) - (2 - 2*s)).abs().max():.3e}   (tolerance 1e-5)")

fig, ax = plt.subplots(figsize=(4.6, 3.4))
grid = torch.linspace(-1, 1, 200)
ax.plot(grid, distance_from_similarity(grid), color="#2a78d6")
ax.scatter(s[:300], d[:300], s=6, color="#eb6834", alpha=0.5, zorder=3)
for x, y, t in [(1, 0, "identical"), (0, 2**0.5, "orthogonal"), (-1, 2, "antipodal")]:
    ax.scatter([x], [y], s=60, color="#0b0b0b", zorder=4)
    ax.annotate(t, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("cosine similarity  s"); ax.set_ylabel("Euclidean distance  d")
ax.set_title("d = sqrt(2 - 2s)   — monotone decreasing")
plt.show()
""")

# =============================================================================
md(r"""
---
## 3 · The encoder $f_\theta$

```
resnet18 → global average pool → Linear(512 → d) → BatchNorm1d(d) → L2 normalise
```

The poster draws two towers. **There is only one module, called twice.** Shared weights
are not a constraint bolted onto the architecture — they *are* the architecture. Two
independently-parameterised towers could map the same face to two unrelated points and
the similarity between them would mean nothing.
""")

code(r"""
from src.models.encoder import Encoder

model = Encoder(backbone="resnet18", embedding_dim=128, normalize=True).to(DEVICE).eval()
print(model.summary())

# "Two branches" == the same module applied twice.
x1 = torch.randn(4, 3, 112, 112, device=DEVICE)
x2 = torch.randn(4, 3, 112, 112, device=DEVICE)
with torch.no_grad():
    za, zb = model(x1), model(x2)

print(f"\nembedding shape {tuple(za.shape)}, norms {za.norm(dim=1)[:4].tolist()}")
print(f"similarity of the two branches' outputs: {cosine_similarity(za, zb).tolist()}")
""")

code(r"""
# 🎛 KNOB — what does turning OFF normalisation do? (This is experiment E6.)
unnormalised = Encoder(embedding_dim=128, normalize=False).to(DEVICE).eval()
with torch.no_grad():
    raw = unnormalised(x1)
print("normalize=False -> norms:", [f"{v:.2f}" for v in raw.norm(dim=1).tolist()])
print("normalize=True  -> norms: all 1.0")
print("\nWithout normalisation the similarity scale drifts as ||z|| grows during")
print("training, so a fixed threshold means something different at every epoch.")
""")

# =============================================================================
md(r"""
---
## 4 · The three losses

### 4.1 Contrastive (Hadsell et al., 2006)

$$L = y\,d^2 + (1-y)\max(0, m-d)^2$$

* $y=1$: loss is $d^2$, minimised only at $d=0$. Positives are pulled with **no floor**.
* $y=0$: loss is $(m-d)^2$ **only while $d<m$**. Past the margin a negative contributes
  exactly zero loss *and* zero gradient.

Because embeddings are normalised, $d\in[0,2]$, so any $m>2$ is unsatisfiable.
""")

code(r"""
MARGIN = 1.0              # 🎛 KNOB: E2 sweeps 0.25 / 0.5 / 1.0 / 1.5. Try 0.25.

from src.losses.contrastive import contrastive_loss_from_distances

# The plan §12 gate vectors:
for dist, y, expected in [(0.5, 1, 0.25), (0.5, 0, 0.25), (1.2, 0, 0.0), (0.0, 1, 0.0)]:
    got = contrastive_loss_from_distances(torch.tensor([dist]), torch.tensor([float(y)]), 1.0)
    print(f"  d={dist}, y={y}  ->  {got.item():.4f}   (expected {expected})")

grid = torch.linspace(0, 2, 300)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(grid, contrastive_loss_from_distances(grid, torch.ones_like(grid), MARGIN),
        color="#2a78d6", label="positive pair (y=1)")
ax.plot(grid, contrastive_loss_from_distances(grid, torch.zeros_like(grid), MARGIN),
        color="#eb6834", label="negative pair (y=0)")
ax.axvline(MARGIN, color="#0b0b0b", ls="--", lw=1.2)
ax.annotate(f"margin m={MARGIN}\nnegatives beyond here\ncontribute NOTHING",
            xy=(MARGIN, 0.5), xytext=(8, 0), textcoords="offset points", fontsize=8)
ax.set_xlabel("distance d"); ax.set_ylabel("loss"); ax.legend()
ax.set_title("Contrastive loss")
plt.show()
""")

md(r"""
### 4.2 Triplet (Schroff et al., 2015 — FaceNet)

$$L = \max(0,\; d_{ap} - d_{an} + \alpha)$$

The contrast with contrastive loss is the whole reason both are here. Contrastive fixes
**absolute** targets (positives → 0, negatives ≥ m). Triplet fixes only a **relative
ordering**: the positive must be closer than the negative by $\alpha$, and where either
sits in absolute terms is free. That is weaker — and usually better, because identities
differ in how tightly they can cluster, and forcing them all to one radius wastes capacity.
""")

code(r"""
ALPHA = 0.2               # 🎛 KNOB: E3 sweeps 0.1 / 0.2 / 0.4

from src.losses.triplet import triplet_loss_from_distances

for d_ap, d_an, expected in [(0.3, 0.9, 0.0), (0.8, 0.9, 0.1)]:
    got = triplet_loss_from_distances(torch.tensor([d_ap]), torch.tensor([d_an]), 0.2)
    print(f"  d_ap={d_ap}, d_an={d_an}  ->  {got.item():.4f}   (expected {expected})")

d_an_grid = torch.linspace(0, 2, 300)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for d_ap, color in [(0.3, "#2a78d6"), (0.8, "#eb6834"), (1.2, "#1baf7a")]:
    loss = triplet_loss_from_distances(torch.full_like(d_an_grid, d_ap), d_an_grid, ALPHA)
    ax.plot(d_an_grid, loss, color=color, label=f"d_ap = {d_ap}")
    ax.axvline(d_ap + ALPHA, color=color, ls=":", lw=1)
ax.set_xlabel("d(anchor, negative)"); ax.set_ylabel("loss"); ax.legend()
ax.set_title(f"Triplet loss (α={ALPHA}) — zero once d_an > d_ap + α")
plt.show()
""")

md(r"""
### 4.3 InfoNCE (van den Oord et al., 2018)

$$L_i = -\log \frac{\exp(s(z_i,z_{i^+})/\tau)}{\sum_{j\in B,\, j\neq i}\exp(s(z_i,z_j)/\tau)}$$

Two details are load-bearing:

**The denominator includes the positive.** It runs over every $j \neq i$, not over
negatives only. That makes the expression a proper softmax cross-entropy, so the ratio
is bounded in $(0,1]$ and the loss in $[0,\infty)$. Excluding the positive would let the
ratio exceed 1 and the loss go negative — unbounded below, and the optimiser would chase
it forever. It changes the loss *value* but not its argmin.

**Only the diagonal is masked**, not all same-identity entries. With K=4, anchor *i* has
3 same-identity images present; the 2 that are not the chosen positive are treated as
negatives. That is standard InfoNCE — "identify the true match among competing
candidates". `supcon=True` is the flag that changes it.
""")

code(r"""
import math
from src.losses.infonce import InfoNCELoss

# The plan §12 gate vector, built geometrically:
#   anchor and positive are the same unit vector (s=1); negative is orthogonal (s=0).
z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
labels = torch.tensor([0, 0, 1])
loss, _ = InfoNCELoss(temperature=1.0)(z, labels)

expected = -math.log(math.e / (math.e + 1.0))
print(f"  computed {loss.item():.8f}")
print(f"  expected {expected:.8f}   = -log(e/(e+1))")
print(f"  |diff|   {abs(loss.item()-expected):.2e}\n")
print("Note it is NOT zero even though the positive is a perfect match --")
print("that floor is exactly what including the positive in the denominator buys.")
""")

code(r"""
TAU = 0.07                # 🎛 KNOB: E4 sweeps 0.03 … 0.5. Try 0.5 and watch the gradient flatten.

# Temperature controls how sharply the loss focuses on the HARDEST negatives.
sims = torch.linspace(-1, 1, 300)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for tau, color in [(0.03, "#2a78d6"), (0.07, "#eb6834"), (0.2, "#1baf7a"), (0.5, "#eda100")]:
    weights = torch.softmax(sims / tau, dim=0)
    ax.plot(sims, weights / weights.max(), color=color, label=f"τ = {tau}")
ax.set_xlabel("similarity of a negative to the anchor")
ax.set_ylabel("relative gradient weight")
ax.set_title("Small τ concentrates gradient on the hardest negatives")
ax.legend()
plt.show()
""")

code(r"""
# The N x N similarity matrix -- poster panel 5, step 4 -- on RANDOM embeddings.
# Compare this with the trained version in §7.4.
torch.manual_seed(0)
labels_demo = torch.arange(12).repeat_interleave(4)
z_demo = l2_normalize(torch.randn(48, 64))

from src.viz.style import SIMILARITY_CMAP
fig, ax = plt.subplots(figsize=(4.4, 3.8))
im = ax.imshow(cosine_similarity_matrix(z_demo), cmap=SIMILARITY_CMAP, vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, fraction=0.046).set_label("cosine similarity", fontsize=8)
ax.set_title("Untrained: no block structure"); ax.grid(False)
plt.show()
""")

# =============================================================================
md(r"""
---
## 5 · Mining — the part that matters more than the margin

Once training is underway, most randomly-drawn triplets already satisfy
$d_{ap}+\alpha < d_{an}$, so they produce **zero loss and zero gradient** — while still
costing a full forward and backward pass. Random mining therefore spends most of its
compute on triplets that teach nothing.

* `random` — uniform positive, uniform negative. The control.
* `semi-hard` — negatives with $d_{ap} < d_{an} < d_{ap}+\alpha$: further than the
  positive, but still inside the margin. Nonzero yet *bounded* gradient.
* `batch-hard` — hardest positive, hardest negative per anchor. Strongest signal, most
  sensitive to label noise (a mislabelled image is exactly what "hardest" selects).

Why not always take the hardest negative? Early in training the hardest negatives are the
model's worst mistakes; pulling hard on them collapses the embedding. Semi-hard sits
deliberately between *useless* and *destructive*.
""")

code(r"""
from src.losses.miners import mine, MINERS
from src.losses.triplet import TripletLoss

torch.manual_seed(0)
labels_m = torch.arange(16).repeat_interleave(4)
z_m = l2_normalize(torch.randn(64, 32))
distances = euclidean_distance_matrix(z_m)

for miner in MINERS:
    a, p, n = mine(z_m, labels_m, miner, margin=0.2)
    d_ap, d_an = distances[a, p], distances[a, n]
    loss, logs = TripletLoss(margin=0.2, miner=miner)(z_m, labels_m)
    print(f"{miner:11s}  d_ap {d_ap.mean():.3f}  d_an {d_an.mean():.3f}  "
          f"active {logs['active_fraction']:.3f}  loss {loss.item():.4f}")
""")

code(r"""
# The semi-hard band, drawn. Each anchor's chosen negative must land in the shaded strip.
ALPHA_VIZ = 0.3           # 🎛 KNOB

a, p, n = mine(z_m, labels_m, "semi-hard", margin=ALPHA_VIZ)
d_ap, d_an = distances[a, p].numpy(), distances[a, n].numpy()

fig, ax = plt.subplots(figsize=(5.0, 4.0))
lo, hi = 0, 2
ax.fill_between([lo, hi], [lo, hi], [lo + ALPHA_VIZ, hi + ALPHA_VIZ],
                color="#1baf7a", alpha=0.18, label="semi-hard band")
ax.plot([lo, hi], [lo, hi], color="#0b0b0b", lw=1, ls="--")
ax.scatter(d_ap, d_an, s=22, color="#eb6834", edgecolors="white", linewidths=0.5, zorder=3)
ax.annotate("d_an = d_ap  (below this the negative\nis CLOSER than the positive)",
            xy=(1.2, 1.2), xytext=(-4, -34), textcoords="offset points", fontsize=7.5)
ax.set_xlabel("d(anchor, positive)"); ax.set_ylabel("d(anchor, negative)")
ax.set_xlim(0.6, 1.8); ax.set_ylim(0.6, 1.8)
ax.set_title(f"semi-hard mining, α={ALPHA_VIZ}"); ax.legend(loc="lower right")
plt.show()
""")

# =============================================================================
md(r"""
---
## 6 · Training — watch it learn

This runs a **small** training job live (a handful of identities, a few epochs) so you can
watch the numbers move without waiting. The full runs are the ones in `results/`.

🎛 Change `LOSS_NAME`, `N_IDENTITIES`, `EPOCHS` and re-run.
""")

code(r"""
LOSS_NAME     = "infonce"    # 🎛 "contrastive" | "triplet" | "infonce"
N_IDENTITIES  = 40           # 🎛 more identities = harder
EPOCHS        = 12           # 🎛
P_IDS, K_IMGS = 8, 4

from torch.utils.data import DataLoader
from src.engine.train import build_loss

loss_cfg = {"contrastive": {"name": "contrastive", "margin": 1.0},
            "triplet":     {"name": "triplet", "margin": 0.2, "miner": "semi-hard"},
            "infonce":     {"name": "infonce", "temperature": 0.07}}[LOSS_NAME]

demo_set = CelebAIdentityDataset(
    identities=splits.train, transform=build_transform("basic", train=True),
    max_identities=N_IDENTITIES,
)
demo_sampler = PKSampler(demo_set.labels, p=P_IDS, k=K_IMGS, seed=0)
demo_loader = DataLoader(demo_set, batch_sampler=demo_sampler, num_workers=4)

demo_model = Encoder(embedding_dim=128).to(DEVICE)
criterion = build_loss(loss_cfg)
optimizer = torch.optim.AdamW(demo_model.parameters(), lr=3e-4, weight_decay=1e-4)

history = []
for epoch in range(EPOCHS):
    demo_sampler.set_epoch(epoch)
    demo_model.train()
    epoch_loss, epoch_logs, steps = 0.0, {}, 0
    for images, labels_b in demo_loader:
        images, labels_b = images.to(DEVICE), labels_b.to(DEVICE)
        loss, logs = criterion(demo_model(images), labels_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(demo_model.parameters(), 5.0)
        optimizer.step()
        epoch_loss += loss.item()
        for k, v in logs.items():
            epoch_logs[k] = epoch_logs.get(k, 0.0) + v
        steps += 1
    row = {"epoch": epoch, "loss": epoch_loss / steps,
           **{k: v / steps for k, v in epoch_logs.items()}}
    history.append(row)
    print(f"epoch {epoch:2d} | " + " | ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"))
""")

code(r"""
import pandas as pd
frame = pd.DataFrame(history)
columns = [c for c in frame.columns if c != "epoch"]

fig, axes = plt.subplots(1, len(columns), figsize=(3.1 * len(columns), 2.8), squeeze=False)
from src.viz.style import CATEGORICAL
for ax, column, color in zip(axes[0], columns, CATEGORICAL):
    ax.plot(frame["epoch"], frame[column], color=color)
    ax.set_title(column, fontsize=9.5, pad=8); ax.set_xlabel("epoch"); ax.margins(y=0.18)
fig.suptitle(f"Live training — {LOSS_NAME}", y=1.06)
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""
---
## 7 · Evaluation

Load the trained baseline and measure it properly. Everything below uses the **test**
identities — never seen during training.
""")

code(r"""
RUN = "results/baseline_infonce"      # 🎛 KNOB: any folder in results/ with a ckpts/best.pt

from src.models.encoder import build_encoder
from src.data.celeba import CelebAPairDataset
from src.engine.evaluate import embed_dataset, score_pairs

checkpoint = torch.load(f"{RUN}/ckpts/best.pt", map_location=DEVICE, weights_only=False)
trained = build_encoder(checkpoint["config"]["model"]).to(DEVICE)
trained.load_state_dict(checkpoint["model"]); trained.eval()
print(f"loaded {RUN} (epoch {checkpoint['epoch']}, val AUC {checkpoint.get('auc', float('nan')):.4f})")

pair_set = CelebAPairDataset(test_pairs, build_transform(train=False))
embeddings = embed_dataset(trained, pair_set, DEVICE, num_workers=8)
scores = score_pairs(embeddings, pair_set.pair_indices)
y = np.asarray(test_labels)
print(f"scored {len(scores):,} test pairs")
""")

md(r"""
### 7.1 The decision: one threshold (poster panel 6)

Everything in verification reduces to `s > t → Accept`. The **overlap** between the two
histograms is precisely the set of pairs no threshold can get right — and its area is
what the ROC curve integrates.
""")

code(r"""
from src.metrics.verification import verification_metrics
from src.viz.histograms import plot_similarity_histograms

metrics = verification_metrics(scores, y)
for key, value in metrics.items():
    print(f"  {key:24s} {value:.4f}")

plot_similarity_histograms(scores, y, "results/_notebook", threshold=metrics["best_threshold"],
                           title="Test pairs — similarity distributions")
from IPython.display import Image as IPyImage
IPyImage("results/_notebook/v1_similarity_histograms.png")
""")

md(r"""
### 7.2 Why TAR@FAR, not accuracy 🎛

Move the threshold and watch the trade. Accuracy peaks in the middle and hides what a
security-facing system cares about: at a **fixed** false-accept rate, how many genuine
users get in?
""")

code(r"""
from src.metrics.verification import accuracy_at, tar_at_far

thresholds = np.linspace(scores.min(), scores.max(), 200)
accuracies = [accuracy_at(scores, y, t) for t in thresholds]
fars = [(scores[y == 0] > t).mean() for t in thresholds]
tars = [(scores[y == 1] > t).mean() for t in thresholds]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
axes[0].plot(thresholds, accuracies, color="#2a78d6")
axes[0].axvline(metrics["best_threshold"], color="#0b0b0b", ls="--", lw=1)
axes[0].set_xlabel("threshold t"); axes[0].set_title("accuracy", pad=8)

axes[1].plot(fars, tars, color="#eb6834")
axes[1].set_xscale("log"); axes[1].set_xlim(1e-4, 1)
for target in (1e-3, 1e-2):
    tar, _ = tar_at_far(scores, y, target)
    axes[1].scatter([target], [tar], s=60, color="#0b0b0b", zorder=4)
    axes[1].annotate(f"TAR@FAR={target:g}\n= {tar:.3f}", (target, tar),
                     textcoords="offset points", xytext=(8, -14), fontsize=8)
axes[1].set_xlabel("false accept rate (log)"); axes[1].set_ylabel("true accept rate")
axes[1].set_title("ROC — the operating points that matter", pad=8)
plt.tight_layout(); plt.show()
""")

md(r"""
### 7.3 LFW, the 10-fold protocol

The threshold is fit on **9 folds** and tested on the held-out **10th**, rotating. Fitting
it on the test fold itself would tune a free parameter on the data being scored — the
result would include the gain from picking the luckiest threshold for those exact pairs,
and would not transfer. The cell below measures that optimism directly.
""")

code(r"""
from src.data.lfw import load_lfw_pairs, LFWImageDataset, DEFAULT_LFW_ROOT
from src.metrics.verification import lfw_10fold_accuracy, best_threshold

lfw = load_lfw_pairs()
lfw_set = LFWImageDataset(lfw, build_transform(train=False), root=DEFAULT_LFW_ROOT)
lfw_embeddings = embed_dataset(trained, lfw_set, DEVICE, num_workers=8)
lfw_scores = score_pairs(lfw_embeddings, lfw_set.pair_indices)

honest = lfw_10fold_accuracy(lfw_scores, lfw.labels, lfw.folds)
print(f"LFW 10-fold (threshold fit OUT of fold): {honest['mean']:.4f} ± {honest['std']:.4f}")

cheating = [best_threshold(lfw_scores[lfw.folds == f], lfw.labels[lfw.folds == f])[1]
            for f in range(10)]
print(f"LFW if you fit the threshold ON the test fold: {np.mean(cheating):.4f}")
print(f"  -> optimism bought by cheating: {np.mean(cheating) - honest['mean']:+.4f}")
""")

md(r"""
### 7.4 The trained similarity matrix — poster panel 5.4

Compare with the untrained version in §4.3.
""")

code(r"""
test_set = CelebAIdentityDataset(
    identities=splits.test, transform=build_transform(train=False), max_identities=200
)

class _Indexed(torch.utils.data.Dataset):
    def __len__(self): return len(test_set)
    def __getitem__(self, i): return test_set[i][0], i

test_embeddings = embed_dataset(trained, _Indexed(), DEVICE, num_workers=8)
test_labels_arr = test_set.labels

heat_sampler = PKSampler(test_labels_arr, p=12, k=4, seed=0, num_batches=1)
rows = next(iter(heat_sampler))

fig, ax = plt.subplots(figsize=(4.4, 3.8))
im = ax.imshow(cosine_similarity_matrix(test_embeddings[rows]), cmap=SIMILARITY_CMAP, vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, fraction=0.046).set_label("cosine similarity", fontsize=8)
ax.set_title("TRAINED: 4x4 same-identity blocks on the diagonal"); ax.grid(False)
plt.show()
""")

md(r"""
### 7.5 Retrieval — a much harder question

Verification asks one binary question about a pair. Retrieval asks whether the right
identity outranks an entire gallery. A single confusable impostor anywhere in the gallery
costs the query, so Recall@1 degrades far faster than pair accuracy.
""")

code(r"""
from src.metrics.retrieval import recall_at_k

retrieval = recall_at_k(test_embeddings, test_labels_arr)
for key, value in retrieval.items():
    print(f"  {key:14s} {value:.4f}")
print(f"\npair accuracy {metrics['best_accuracy']:.4f}  vs  recall@1 {retrieval['recall@1']:.4f}")
print("Same embedding. Different question.")
""")

md(r"""
### 7.6 Verify two faces (poster panel 6, made tangible)
""")

code(r"""
IMG_A = f"data/celeba/img_align_celeba/{test_pairs[0][0]}"    # 🎛 try any two files
IMG_B = f"data/celeba/img_align_celeba/{test_pairs[0][1]}"

pil_a, pil_b = Image.open(IMG_A), Image.open(IMG_B)
tf = build_transform(train=False)
with torch.no_grad():
    emb = trained(torch.stack([tf(pil_a), tf(pil_b)]).to(DEVICE)).float().cpu()
similarity = cosine_similarity(emb[0:1], emb[1:2]).item()
t = metrics["best_threshold"]
accept = similarity > t

fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.6))
for ax, img, label in zip(axes, (pil_a, pil_b), ("A", "B")):
    ax.imshow(img); ax.set_title(label, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle(f"s = {similarity:.3f}  {'>' if accept else '<'}  t = {t:.3f}", y=1.08)
fig.text(0.5, 0.95, "ACCEPT — same person" if accept else "REJECT — different person",
         ha="center", fontsize=11, fontweight="bold",
         color="#008300" if accept else "#e34948")
plt.show()
""")

# =============================================================================
md(r"""
---
## 8 · Alignment & uniformity — *why*, not just *how well*

Wang & Isola (2020) showed the InfoNCE objective decomposes asymptotically into two
competing terms:

* **alignment** $= \mathbb{E}_{pos}\|\hat z_i - \hat z_j\|^2$ — how tightly positives sit
  together. Lower is better.
* **uniformity** $= \log \mathbb{E}_{i,j} e^{-2\|\hat z_i - \hat z_j\|^2}$ — how evenly
  embeddings spread over the sphere. Lower is better.

The pair is diagnostic because the two failure modes are **opposite**, and each looks
fine on one metric alone:

* **collapse** — everything maps to one point. Alignment is a perfect 0; uniformity is at
  its worst. Alignment alone would call this a success.
* **dispersion** — beautifully spread, but positives are not together. Uniformity great,
  alignment poor.
""")

code(r"""
from src.metrics.align_uniform import align_uniform

real = align_uniform(test_embeddings, test_labels_arr)

collapsed = torch.zeros(400, 64); collapsed[:, 0] = 1.0
scattered = l2_normalize(torch.randn(400, 64))
fake_labels = np.repeat(np.arange(100), 4)

points = {
    "trained model": (real["alignment"], real["uniformity"]),
    "collapsed":     tuple(align_uniform(collapsed, fake_labels).values()),
    "random":        tuple(align_uniform(scattered, fake_labels).values()),
}
for name, (a, u) in points.items():
    print(f"  {name:15s} alignment {a:7.4f}   uniformity {u:8.4f}")

fig, ax = plt.subplots(figsize=(5.0, 4.0))
for (name, (a, u)), color in zip(points.items(), CATEGORICAL):
    ax.scatter(a, u, s=110, color=color, edgecolors="white", linewidths=1.5, zorder=3)
    ax.annotate(name, (a, u), textcoords="offset points", xytext=(9, 4),
                fontsize=8.5, color=color, fontweight="bold")
ax.set_xlabel("alignment  (lower = tighter positives)")
ax.set_ylabel("uniformity  (lower = better spread)")
ax.set_title("Lower-left is good; each axis alone can be fooled")
plt.show()
""")

# =============================================================================
md(r"""
---
## 9 · The silent bug: cross-GPU negatives

This is the most important idea in the lab, and you can see it **without any GPUs**.

For anchor $i$ and any **negative** $j$:

$$\frac{\partial L_i}{\partial z_j} = \frac{1}{\tau}\,p_{ij}\,z_i,
\qquad p_{ij} = \mathrm{softmax}_j(s_{ij}/\tau)$$

which is **nonzero**. Negatives are not passive context — the loss actively pushes them
away, and that push is a real gradient that must reach whichever GPU owns $z_j$.

`torch.distributed.all_gather` returns tensors with **no autograd history**. With it, that
push is silently discarded. Training still runs. The loss still falls. Every curve looks
healthy. But you are optimising small-batch InfoNCE while believing you have large-batch
InfoNCE — which would invalidate experiment E5 entirely.

The cell below simulates both, single-process.
""")

code(r"""
torch.manual_seed(0)
labels_g = torch.arange(16).repeat_interleave(2)
base = l2_normalize(torch.randn(32, 64))

def local_grad(detach_pool: bool):
    # Gradient w.r.t. the OTHER rank's rows, with and without a detached pool.
    z = base.clone().requires_grad_(True)
    mine_, theirs = z[:16], z[16:]
    pool = torch.cat([mine_, theirs.detach() if detach_pool else theirs])
    loss, _ = InfoNCELoss(temperature=0.07, seed=0)(
        mine_, labels_g[:16], pool, labels_g, rank_offset=0)
    loss.backward()
    return z.grad[16:].clone()          # gradient reaching the OTHER rank's rows

correct  = local_grad(detach_pool=False)
detached = local_grad(detach_pool=True)

print(f"gradient reaching the other rank's embeddings")
print(f"  gradient-preserving gather : {correct.abs().sum():.6f}")
print(f"  detached all_gather        : {detached.abs().sum():.6f}   <-- silently ZERO")
print(f"\nNothing errors. Nothing warns. The loss is identical in both cases:")
print("only the gradients differ -- which is why the Phase 4 gate checks GRADIENTS.")
""")

code(r"""
# Does the gate actually catch it? (Requires 3 GPUs; see results/phase_reports/phase_4.md.)
print(open("results/phase_reports/phase_4.md").read()[:1500])
""")

# =============================================================================
md(r"""
---
## 10 · Explore the experiment matrix

Every completed run's numbers, in one frame. Slice it however you like.
""")

code(r"""
import json, glob

rows = []
for path in sorted(glob.glob("results/*/metrics.json")):
    name = Path(path).parent.name
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        continue
    if "internal" not in data:
        continue
    internal, lfw = data["internal"], data.get("lfw", {})
    rows.append({
        "run": name,
        "experiment": name.split("_")[0],
        "lfw": lfw.get("mean", np.nan),
        "lfw_std": lfw.get("std", np.nan),
        "auc": internal.get("auc", np.nan),
        "tar@1e-3": internal.get("tar@far=0.001", np.nan),
        "eer": internal.get("eer", np.nan),
        "recall@1": data.get("retrieval", {}).get("recall@1", np.nan),
        "alignment": data.get("representation", {}).get("alignment", np.nan),
        "uniformity": data.get("representation", {}).get("uniformity", np.nan),
    })

runs = pd.DataFrame(rows).sort_values("lfw", ascending=False).reset_index(drop=True)
print(f"{len(runs)} completed runs")
runs.head(20)
""")

code(r"""
# 🎛 KNOB — pick an experiment: e1 (losses) e2 (margin) e3 (mining) e4 (τ)
#                               e5 (batch) e6 (norm) e7 (aug) e8 (dim)
EXPERIMENT = "e1"

subset = runs[runs["experiment"] == EXPERIMENT].sort_values("lfw", ascending=False)
display(subset)

if len(subset) > 1:
    fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(subset) + 1.4))
    ax.barh(subset["run"], subset["lfw"], xerr=subset["lfw_std"],
            color=CATEGORICAL[0], height=0.6, error_kw={"lw": 1, "ecolor": "#52514e"})
    ax.set_xlabel("LFW accuracy (10-fold mean ± std)")
    ax.set_xlim(max(0.5, subset["lfw"].min() - 0.06), min(1.0, subset["lfw"].max() + 0.03))
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_title(f"{EXPERIMENT.upper()}")
    plt.tight_layout(); plt.show()
""")

code(r"""
# Metric choice changes the story. LFW compresses differences that TAR@FAR and
# Recall@1 show as enormous -- this is the poster's panel-7 claim in one plot.
e1 = runs[runs["experiment"] == "e1"].set_index("run")
if len(e1) >= 2:
    show = ["lfw", "auc", "tar@1e-3", "recall@1"]
    # Colour follows the MODEL, not the bar's rank -- otherwise a metric that
    # orders the three differently would repaint them and the eye would track
    # the wrong thing across panels.
    from src.viz.style import color_for
    palette = {name: color_for(name.replace("e1_", "").replace("_semihard", ""), i)
               for i, name in enumerate(e1.index)}

    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 3.0))
    for ax, column in zip(axes, show):
        ordered = e1[column].sort_values()
        ax.barh(range(len(ordered)), ordered.values,
                color=[palette[n] for n in ordered.index], height=0.6)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels([n.replace("e1_", "") for n in ordered.index], fontsize=8)
        ax.set_title(column, fontsize=10, pad=8); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1)
    fig.suptitle("Same three models, four metrics", y=1.04)
    plt.tight_layout(); plt.show()
""")

md(r"""
---
## Where to go next

* **`results/phase_reports/phase_0.md … phase_6.md`** — what was built and found at each
  stage, including the bugs and the reasoning that caught them.
* **`FINDINGS.md`** — every poster claim, with a verdict, a figure and a number.
* **`docs/plan.md`** §10 — the full experiment matrix and its predicted outcomes.
* **`pytest tests/ -q`** — 190+ tests; `tests/test_losses.py` contains the exact
  formula vectors, and `tests/ddp_equivalence.py` is the critical gate.

Things worth trying yourself:

1. Set `MARGIN = 0.25` in §4.1 and re-run the live training in §6 with `contrastive` —
   does the separation get worse, and does the histogram in §7.1 shift the way you expect?
2. Set `TAU = 0.5` in §4.3 and re-train. What happens to the gap between positive and
   negative similarity?
3. In §6, switch the miner to `random` and watch `active_fraction` collapse.
4. Break something on purpose: pass an un-normalised embedding into the InfoNCE loss and
   see which assertion catches it first.
""")


def main() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": kind,
                "id": f"cell-{index:02d}",
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for index, (kind, source) in enumerate(CELLS)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "face-siamese-lab (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1))
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {OUT} — {len(CELLS)} cells ({n_code} code, {len(CELLS) - n_code} markdown)")


if __name__ == "__main__":
    main()
