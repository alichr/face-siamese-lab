#!/usr/bin/env python3
"""
Siamese face verification in one file — a readable, runnable minimal version.

This is the whole lab (../src/) compressed into one ~1,000-line file you can read top to
bottom in one sitting. Same maths, same bug fixes, no package structure.

    python siamese_minimal.py selftest        # exact loss vectors, ~2 s, no data needed
    python siamese_minimal.py explain         # the key ideas with live numbers
    python siamese_minimal.py demo            # train + evaluate + plot, ~2 min
    python siamese_minimal.py train  --loss infonce --epochs 5
    python siamese_minimal.py eval   --ckpt runs/infonce/model.pt
    python siamese_minimal.py compare --epochs 5     # all three losses, side by side
    python siamese_minimal.py verify A.jpg B.jpg --ckpt runs/infonce/model.pt

Only needs CelebA: `--data ../data/celeba` (the default), containing
`img_align_celeba/` and `identity_CelebA.txt`.

READING ORDER — the code below is ordered by dependency, which is also the
order the ideas build on each other:

    §1  geometry     why cosine and Euclidean are the same thing
    §2  data         identity splits, PK batches (the thing that makes positives exist)
    §3  model        the encoder; "shared weights" is one module called twice
    §4  losses       contrastive, triplet (+3 miners), InfoNCE
    §5  metrics      AUC / TAR@FAR / EER / accuracy, and why the choice matters
    §6  train        the loop
    §7  plots        what to look at
    §8  cli          the commands above

Everything marked "GOTCHA" is a bug that actually happened in the full lab and
cost real debugging time. They are kept here because they are the interesting part.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ============================================================================
# §1  GEOMETRY — the identity everything rests on
# ============================================================================
#
# For L2-normalised embeddings:
#
#     d(a,b)^2 = ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b = 2 - 2 s(a,b)
#
# So d is a strictly DECREASING function of s. Thresholding one is thresholding
# the other. Contrastive and triplet are written in d, InfoNCE in s; on the unit
# sphere they rank the same pairs. That is the whole reason the poster can draw
# "cosine similarity" and "Euclidean distance" as two boxes.

EPS = 1e-12


def l2_normalize(z: torch.Tensor) -> torch.Tensor:
    """Project onto the unit sphere: zhat = z / ||z||."""
    return z / z.norm(dim=-1, keepdim=True).clamp_min(EPS)


def safe_sqrt(squared: torch.Tensor) -> torch.Tensor:
    """sqrt with a finite gradient at zero.

    GOTCHA #1 (cost: half a debugging session).
    d/dx sqrt(x) = 1/(2 sqrt(x)) is INFINITE at x=0, and the self-similarity
    diagonal sits at exactly 0. Every loss masks that diagonal out, so its
    upstream gradient is exactly 0 -- and `0 * inf = NaN`, which then poisons the
    entire backward pass. The forward values all look perfectly fine.

    Both halves of the fix are needed:
      - clamp before sqrt  -> the local derivative is large but finite
      - masked_fill after  -> restores exact 0.0 AND blocks gradient flow there
    """
    is_zero = squared <= 0
    return squared.clamp_min(EPS).sqrt().masked_fill(is_zero, 0.0)


def sim_matrix(z: torch.Tensor, other: torch.Tensor | None = None) -> torch.Tensor:
    """Cosine similarity matrix S = Zhat Zhat^T, in [-1, 1]. Poster panel 5.4."""
    a = l2_normalize(z)
    b = a if other is None else l2_normalize(other)
    return a @ b.T


def dist_matrix(z: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance on the unit sphere, via d^2 = 2 - 2s."""
    return safe_sqrt(2.0 - 2.0 * sim_matrix(z))


# ============================================================================
# §2  DATA — identity splits and PK batches
# ============================================================================


def load_identities(root: Path) -> dict[int, list[str]]:
    """Parse identity_CelebA.txt into {identity: [filenames]}."""
    path = root / "identity_CelebA.txt"
    if not path.exists():
        raise SystemExit(
            f"CelebA not found at {root}\n"
            f"Expected {root}/img_align_celeba/ and {root}/identity_CelebA.txt\n"
            "Point --data at your CelebA copy, or fetch it with:\n"
            "  python -c \"import torchvision;"
            "torchvision.datasets.CelebA(root='data', download=True)\"  # needs gdown"
        )
    grouped: dict[int, list[str]] = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            filename, identity = line.split()
            grouped[int(identity)].append(filename)
    return {i: sorted(f) for i, f in grouped.items()}


def split_identities(by_id: dict[int, list[str]], n_train: int, n_val: int, seed: int = 0):
    """Split by IDENTITY, never by image.

    This is the single most important design decision in the project. If the same
    person appeared in both train and test, the model could answer "same person?"
    from memorised identity rather than any transferable notion of similarity --
    and the reported accuracy would collapse on a face it had never seen.

    Because that failure makes results look BETTER, it deserves an assertion.
    """
    ids = np.random.default_rng(seed).permutation(sorted(by_id))
    train, val, test = ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

    assert not (set(train) & set(val)), "IDENTITY LEAK: train n val"
    assert not (set(train) & set(test)), "IDENTITY LEAK: train n test"
    assert not (set(val) & set(test)), "IDENTITY LEAK: val n test"

    return [int(i) for i in train], [int(i) for i in val], [int(i) for i in test]


def build_transform(train: bool, augment: str = "basic"):
    """resize 128 -> 112 crop, normalised to roughly [-1, 1].

    Note `none` still random-crops when training -- it means "crop only", not
    "no randomness". That is why an overfit run bottoms out near 0.06, not 0.
    """
    if not train:
        return transforms.Compose([
            transforms.Resize(128), transforms.CenterCrop(112),
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    steps = [transforms.Resize(128), transforms.RandomCrop(112)]
    if augment in ("basic", "strong"):
        steps.append(transforms.RandomHorizontalFlip())
    if augment == "strong":
        steps += [
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.5),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    if augment == "strong":
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)))
    return transforms.Compose(steps)


class FaceDataset(Dataset):
    """CelebA images for a set of identities, yielding (image, contiguous_label)."""

    def __init__(self, by_id, identities, transform, root: Path):
        self.root, self.transform = Path(root) / "img_align_celeba", transform
        self.files, self.labels = [], []
        for new_label, identity in enumerate(sorted(identities)):
            for filename in by_id[identity]:
                self.files.append(filename)
                self.labels.append(new_label)
        self.labels = np.array(self.labels)
        self.n_identities = len(set(self.labels.tolist()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image = Image.open(self.root / self.files[i]).convert("RGB")
        return self.transform(image), int(self.labels[i])


class PKSampler(torch.utils.data.Sampler):
    """Every batch is exactly P identities x K images.

    This is what makes in-batch positives EXIST. With K=1 a batch is P singleton
    identities and there are no positive pairs at all -- none of the three losses
    would have anything to pull together.

    Identities with fewer than K images are dropped rather than sampled with
    replacement: a duplicated image is a positive pair at distance exactly 0,
    which is free, uninformative loss.
    """

    def __init__(self, labels, p=32, k=4, batches=100, seed=0):
        assert k >= 2, "k must be >= 2 or there are no positives"
        self.p, self.k, self.batches, self.seed, self.epoch = p, k, batches, seed, 0
        pools = collections.defaultdict(list)
        for index, label in enumerate(labels):
            pools[int(label)].append(index)
        self.pools = {i: np.array(v) for i, v in pools.items() if len(v) >= k}
        self.ids = np.array(sorted(self.pools))
        self.dropped = len(pools) - len(self.pools)
        assert len(self.ids) >= p, f"only {len(self.ids)} usable identities, need {p}"
        self.batch_size = p * k

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        order, cursor = rng.permutation(self.ids), 0
        for _ in range(self.batches):
            if cursor + self.p > len(order):
                order, cursor = rng.permutation(self.ids), 0
            batch = []
            for identity in order[cursor:cursor + self.p]:
                batch += [int(x) for x in rng.choice(self.pools[identity], self.k, replace=False)]
            cursor += self.p
            yield batch


def make_pairs(by_id, identities, n_per_class=1500, seed=0):
    """Fixed verification pairs: n positive (same id) + n negative (different ids)."""
    rng = np.random.default_rng(seed)
    eligible = [i for i in identities if len(by_id[i]) >= 2]
    positives, negatives = set(), set()

    while len(positives) < n_per_class:
        files = by_id[eligible[rng.integers(len(eligible))]]
        a, b = rng.choice(len(files), 2, replace=False)
        positives.add(tuple(sorted((files[a], files[b]))))

    ids = list(identities)
    while len(negatives) < n_per_class:
        i, j = rng.choice(len(ids), 2, replace=False)
        fa, fb = by_id[ids[i]], by_id[ids[j]]
        negatives.add(tuple(sorted((fa[rng.integers(len(fa))], fb[rng.integers(len(fb))]))))

    pairs = sorted(positives) + sorted(negatives)
    labels = np.array([1] * len(positives) + [0] * len(negatives))
    return pairs, labels


# ============================================================================
# §3  MODEL — the encoder
# ============================================================================


class Encoder(nn.Module):
    """resnet18 -> global avg pool -> Linear(512->d) -> BatchNorm1d -> L2 norm.

    The poster draws two towers. There is ONE module, called twice. Shared
    weights are not a constraint bolted on -- they are the architecture. Two
    separately-parameterised towers could map the same face to two unrelated
    points and the similarity between them would mean nothing.
    """

    def __init__(self, dim=128, normalize=True):
        super().__init__()
        net = models.resnet18(weights=None)   # no pretraining: keeps loss comparisons clean
        net.fc = nn.Identity()
        self.backbone, self.head, self.bn = net, nn.Linear(512, dim), nn.BatchNorm1d(dim)
        self.dim, self.normalize = dim, normalize

    def forward(self, x):
        z = self.bn(self.head(self.backbone(x)))
        return l2_normalize(z) if self.normalize else z


# ============================================================================
# §4  LOSSES
# ============================================================================


def contrastive_loss(z, labels, margin=1.0, generator=None):
    """L = y*d^2 + (1-y)*max(0, m-d)^2      (Hadsell et al. 2006, poster 4a)

      y=1 -> loss is d^2, minimised only at d=0. Positives pulled with NO floor.
      y=0 -> loss is (m-d)^2 but ONLY while d<m. Past the margin a negative
             contributes exactly zero loss and zero gradient.

    Normalised embeddings give d in [0,2], so m must be <= 2.

    The negative subsampling below is not cosmetic: a P=32,K=4 batch holds 192
    positive pairs and 7,936 negative ones. Left alone, the negative term
    dominates the gradient ~41:1.
    """
    d = dist_matrix(z)
    n = len(labels)
    same = labels[:, None] == labels[None, :]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=z.device), 1)

    pi, pj = (same & upper).nonzero(as_tuple=True)
    ni, nj = ((~same) & upper).nonzero(as_tuple=True)
    if not len(pi) or not len(ni):
        return z.sum() * 0.0, {}

    keep = torch.randperm(len(ni), generator=generator)[:len(pi)].to(ni.device)
    ni, nj = ni[keep], nj[keep]

    d_pos, d_neg = d[pi, pj], d[ni, nj]
    loss = torch.cat([d_pos.pow(2), (margin - d_neg).clamp_min(0).pow(2)]).mean()
    return loss, {
        "d_pos": d_pos.mean().item(),
        "d_neg": d_neg.mean().item(),
        "neg_in_margin": (d_neg < margin).float().mean().item(),
    }


def mine_triplets(z, labels, miner="semi-hard", margin=0.2, generator=None):
    """Pick one (anchor, positive, negative) per anchor.

    The miner matters far more than the margin -- in the full lab the miner
    changed LFW by 0.11 while the margin changed it by 0.005, a 22x ratio.

      random      uniform choice. Most triplets are already satisfied, so they
                  give zero gradient while still costing a full forward+backward.
      semi-hard   d_ap < d_an < d_ap + margin: further than the positive, but
                  still inside the margin. Nonzero yet BOUNDED gradient.
      batch-hard  hardest positive, hardest negative. Strongest signal -- and in
                  the full lab, the WORST of the three (6-11 LFW points below
                  random) because "hardest" is exactly what a mislabelled image
                  wins, and it partially collapses the embedding.
    """
    d = dist_matrix(z)
    n = len(labels)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos_mask, neg_mask = same & ~eye, ~same

    valid = pos_mask.any(1) & neg_mask.any(1)
    if not valid.any():
        empty = torch.empty(0, dtype=torch.long, device=z.device)
        return empty, empty, empty

    def pick_random(mask):
        noise = torch.rand(mask.shape, generator=generator).to(z.device)
        return (noise * mask).argmax(1)

    if miner == "batch-hard":
        pos = d.masked_fill(~pos_mask, float("-inf")).argmax(1)   # furthest positive
        neg = d.masked_fill(~neg_mask, float("inf")).argmin(1)    # closest negative
    elif miner == "random":
        pos, neg = pick_random(pos_mask), pick_random(neg_mask)
    elif miner == "semi-hard":
        pos = pick_random(pos_mask)
        d_ap = d.gather(1, pos[:, None])
        band = neg_mask & (d > d_ap) & (d < d_ap + margin)
        # Fallback when the band is empty: the closest negative that still violates.
        violating = neg_mask & (d < d_ap + margin)
        source = torch.where(violating.any(1, keepdim=True), violating, neg_mask)
        fallback = d.masked_fill(~source, float("inf")).argmin(1)
        neg = torch.where(band.any(1), pick_random(band), fallback)
    else:
        raise ValueError(f"unknown miner {miner!r}")

    idx = torch.arange(n, device=z.device)
    return idx[valid], pos[valid], neg[valid]


def triplet_loss(z, labels, margin=0.2, miner="semi-hard", generator=None):
    """L = max(0, d_ap - d_an + alpha)      (Schroff et al. 2015, poster 4b)

    Contrastive fixes ABSOLUTE targets (positives->0, negatives>=m). Triplet
    fixes only a RELATIVE ordering, which is weaker and usually better: identities
    differ in how tightly they can cluster, and forcing one radius wastes capacity.

    `active_fraction` is the most instructive number here -- it is the share of
    the batch still producing any gradient at all.
    """
    a, p, n = mine_triplets(z, labels, miner, margin, generator)
    if not len(a):
        return z.sum() * 0.0, {}

    d = dist_matrix(z)
    per_triplet = (d[a, p] - d[a, n] + margin).clamp_min(0)

    # Mean over ALL mined triplets, not just active ones: averaging over actives
    # would rescale the gradient as the active fraction falls, hiding the effect.
    return per_triplet.mean(), {
        "active_fraction": (per_triplet > 0).float().mean().item(),
        "d_ap": d[a, p].mean().item(),
        "d_an": d[a, n].mean().item(),
    }


def infonce_loss(z, labels, tau=0.07, supcon=False, generator=None):
    """L_i = -log[ exp(s_i,i+ /t) / sum_{j != i} exp(s_ij /t) ]   (Oord 2018, poster 4c)

    Two details are load-bearing:

    THE DENOMINATOR INCLUDES THE POSITIVE. It runs over every j != i, not over
    negatives only. That makes this a proper softmax cross-entropy, so the ratio
    is bounded in (0,1] and the loss in [0, inf). Drop the positive and the ratio
    can exceed 1, the loss goes negative, and the optimiser chases it forever.
    It changes the loss VALUE but not its argmin.

    ONLY THE DIAGONAL IS MASKED, not all same-identity entries. With K=4 an
    anchor has 3 same-identity images present; the 2 that are not the chosen
    positive count as negatives. That is standard InfoNCE -- "identify the true
    match among competing candidates". `supcon=True` changes it.
    """
    n = len(labels)
    s = sim_matrix(z)
    logits = s / tau

    rows = torch.arange(n, device=z.device)
    logits[rows, rows] = float("-inf")            # mask self-similarity

    pos_mask = (labels[:, None] == labels[None, :]).clone()
    pos_mask[rows, rows] = False

    valid = pos_mask.any(1)
    if not valid.any():
        return z.sum() * 0.0, {}

    log_denominator = torch.logsumexp(logits, dim=1)

    if supcon:
        log_probs = logits - log_denominator[:, None]
        # GOTCHA #2: use masked_fill, NOT `log_probs * pos_mask`. `logits` holds
        # -inf on the masked diagonal and `-inf * 0` is NaN, not 0. Masking by
        # multiplication is only safe when the tensor is finite everywhere.
        per_anchor = -log_probs.masked_fill(~pos_mask, 0.0).sum(1) / pos_mask.sum(1).clamp_min(1)
    else:
        noise = torch.rand(pos_mask.shape, generator=generator).to(z.device)
        positive = (noise * pos_mask).argmax(1)
        per_anchor = -(logits[rows, positive] - log_denominator)

    with torch.no_grad():
        neg_mask = ~(labels[:, None] == labels[None, :])
        s_pos = (s * pos_mask).sum(1) / pos_mask.sum(1).clamp_min(1)
        s_neg = (s * neg_mask).sum(1) / neg_mask.sum(1).clamp_min(1)

    return per_anchor[valid].mean(), {
        "s_pos": s_pos[valid].mean().item(),
        "s_neg": s_neg[valid].mean().item(),
        "gap": (s_pos - s_neg)[valid].mean().item(),
    }


def build_loss(name, margin=None, miner="semi-hard", tau=0.07, seed=0):
    """Return a `fn(embeddings, labels) -> (loss, logs)`."""
    generator = torch.Generator().manual_seed(seed)
    if name == "contrastive":
        m = 1.0 if margin is None else margin
        return lambda z, y: contrastive_loss(z, y, m, generator)
    if name == "triplet":
        m = 0.2 if margin is None else margin
        return lambda z, y: triplet_loss(z, y, m, miner, generator)
    if name == "infonce":
        return lambda z, y: infonce_loss(z, y, tau, False, generator)
    raise ValueError(f"unknown loss {name!r}")


# ============================================================================
# §5  METRICS — and why the choice of metric matters more than you think
# ============================================================================


def verification_metrics(scores, labels, fars=(1e-3, 1e-2)):
    """AUC, EER, best-threshold accuracy, and TAR at fixed FAR.

    Why TAR@FAR rather than accuracy: accuracy at a 50/50 pair balance hides the
    low-false-accept regime a real system operates in. Two systems differing only
    in 0.5% of impostors scoring like genuine users land within 5% on accuracy
    but >10 points apart on TAR@FAR=1e-3.

    In the full lab the three losses spanned 0.072 on accuracy and 0.372 on
    TAR@FAR=1e-3 -- a 5.8x wider separation, from the same models.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    far, tar, thresholds = roc_curve(labels, scores)

    # EER: where FAR meets FRR.
    frr = 1 - tar
    eer_index = int(np.argmin(np.abs(far - frr)))

    # Best accuracy over a threshold sweep.
    order = np.argsort(scores)
    y = labels[order]
    n_pos = int(y.sum())
    correct = np.cumsum(1 - y) + (n_pos - np.cumsum(y))
    best = int(np.argmax(correct))

    out = {
        "auc": float(roc_auc_score(labels, scores)),
        "eer": float((far[eer_index] + frr[eer_index]) / 2),
        "accuracy": float(correct[best] / len(y)),
        "threshold": float(scores[order][best]),
        "s_pos": float(scores[labels == 1].mean()),
        "s_neg": float(scores[labels == 0].mean()),
    }
    for target in fars:
        feasible = far <= target
        out[f"tar@far={target:g}"] = float(tar[feasible].max()) if feasible.any() else 0.0
    return out


def alignment_uniformity(z, labels):
    """Wang & Isola (2020) — WHY a model behaves as it does, not just how well.

    alignment  = E_pos ||zi - zj||^2         lower = tighter positives
    uniformity = log E exp(-2 ||zi - zj||^2) lower = better spread

    The pair is diagnostic because the failure modes are opposite:
      collapse   -> perfect alignment (0), terrible uniformity (~0). Alignment
                    alone would call this a success. It is the worst outcome.
      dispersion -> great uniformity, poor alignment.
    """
    z = l2_normalize(z)
    labels = np.asarray(labels)

    left, right = [], []
    for identity in np.unique(labels):
        rows = np.flatnonzero(labels == identity)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left.append(rows[i]); right.append(rows[j])
    align = (z[left] - z[right]).pow(2).sum(1).mean().item() if left else float("nan")

    sample = z[:2048]
    sq = torch.cdist(sample, sample).pow(2)
    off = ~torch.eye(len(sample), dtype=torch.bool)
    values = (-2.0 * sq)[off]
    uniform = (torch.logsumexp(values, 0) - math.log(values.numel())).item()
    return {"alignment": align, "uniformity": uniform}


@torch.no_grad()
def embed_files(model, files, root: Path, device, batch=256, workers=6):
    """Encode a list of filenames, deduplicated, in eval mode."""
    unique = sorted(set(files))
    index = {f: i for i, f in enumerate(unique)}
    transform = build_transform(train=False)

    class _DS(Dataset):
        def __len__(self): return len(unique)
        def __getitem__(self, i):
            img = Image.open(Path(root) / "img_align_celeba" / unique[i]).convert("RGB")
            return transform(img), i

    was_training = model.training
    model.eval()
    out = None
    for images, rows in DataLoader(_DS(), batch_size=batch, num_workers=workers):
        z = model(images.to(device)).float().cpu()
        if out is None:
            out = torch.empty(len(unique), z.shape[1])
        out[rows] = z
    if was_training:
        model.train()
    return out, index


@torch.no_grad()
def evaluate(model, pairs, labels, root: Path, device, workers=6):
    """Score a pair list and return the metric dict."""
    embeddings, index = embed_files(model, [f for p in pairs for f in p], root, device,
                                    workers=workers)
    z = l2_normalize(embeddings)
    a = z[[index[p[0]] for p in pairs]]
    b = z[[index[p[1]] for p in pairs]]
    scores = (a * b).sum(1).numpy()
    return verification_metrics(scores, labels), scores


# ============================================================================
# §6  TRAINING
# ============================================================================


def train(args) -> dict:
    """The loop: sample PK batch -> encode -> loss -> backprop -> repeat."""
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data)

    by_id = load_identities(root)
    train_ids, val_ids, _ = split_identities(by_id, args.train_ids, args.val_ids, args.seed)

    train_set = FaceDataset(by_id, train_ids, build_transform(True, args.augment), root)
    sampler = PKSampler(train_set.labels, args.p, args.k, args.batches, args.seed)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)

    val_pairs, val_labels = make_pairs(by_id, val_ids, args.val_pairs, args.seed)

    print(f"train  {len(train_set):,} images / {train_set.n_identities:,} identities "
          f"({sampler.dropped} dropped, <K images)")
    print(f"val    {len(val_pairs):,} pairs from {len(val_ids):,} UNSEEN identities")
    print(f"batch  P={args.p} x K={args.k} = {sampler.batch_size}, "
          f"{args.batches} batches/epoch on {device}")

    model = Encoder(args.dim, normalize=not args.no_normalize).to(device)
    criterion = build_loss(args.loss, args.margin, args.miner, args.tau, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out) / args.loss
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best_auc = [], -1.0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        started, totals, steps = time.time(), collections.defaultdict(float), 0

        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            loss, logs = criterion(model(images), labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            totals["loss"] += loss.item()
            for key, value in logs.items():
                totals[key] += value
            steps += 1

        row = {"epoch": epoch, **{k: v / steps for k, v in totals.items()},
               "seconds": time.time() - started}
        metrics, _ = evaluate(model, val_pairs, val_labels, root, device, args.workers)
        row.update({f"val_{k}": v for k, v in metrics.items()})
        history.append(row)

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save({"model": model.state_dict(), "dim": args.dim,
                        "normalize": not args.no_normalize, "args": vars(args),
                        "epoch": epoch, "auc": best_auc}, out_dir / "model.pt")

        diagnostics = " ".join(f"{k} {v:.3f}" for k, v in row.items()
                               if k not in ("epoch", "seconds", "loss")
                               and not k.startswith("val_"))
        print(f"epoch {epoch:2d} | loss {row['loss']:.4f} | {diagnostics} "
              f"| val AUC {metrics['auc']:.4f} acc {metrics['accuracy']:.4f} "
              f"| {row['seconds']:.0f}s", flush=True)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest val AUC {best_auc:.4f} -> {out_dir}/model.pt")
    return {"history": history, "best_auc": best_auc, "dir": out_dir}


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Encoder(checkpoint["dim"], checkpoint["normalize"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


# ============================================================================
# §7  PLOTS
# ============================================================================


def plot_all(model, scores, labels, metrics, history, out_dir: Path, root: Path, device):
    """Four panels: score histograms, ROC, similarity matrix, training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    # (a) the decision -- poster panel 6. The overlap is what no threshold can fix.
    ax = axes[0, 0]
    bins = np.linspace(min(scores.min(), -0.3), 1.0, 55)
    ax.hist(scores[labels == 0], bins=bins, color=ORANGE, alpha=0.75, label="different person")
    ax.hist(scores[labels == 1], bins=bins, color=BLUE, alpha=0.75, label="same person")
    ax.axvline(metrics["threshold"], color="black", ls="--", lw=1.3)
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("pairs"); ax.legend(frameon=False)
    ax.set_title(f"decision: t={metrics['threshold']:.3f}, acc={metrics['accuracy']:.3f}")

    # (b) ROC on a LOG FAR axis -- a linear axis hides the region that matters.
    ax = axes[0, 1]
    far, tar, _ = roc_curve(labels, scores)
    ax.plot(np.clip(far, 1e-5, None), tar, color=BLUE)
    for target in (1e-3, 1e-2):
        value = metrics[f"tar@far={target:g}"]
        ax.scatter([target], [value], color="black", zorder=4, s=45)
        ax.annotate(f"TAR@{target:g} = {value:.3f}", (target, value), fontsize=8,
                    textcoords="offset points", xytext=(10, 8))
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("false accept rate (log)"); ax.set_ylabel("true accept rate")
    ax.set_title(f"ROC — AUC {metrics['auc']:.4f}, EER {metrics['eer']:.4f}")

    # (c) the batch similarity matrix -- poster panel 5.4, identity-ordered.
    ax = axes[1, 0]
    by_id = load_identities(root)
    _, _, test_ids = split_identities(by_id, 300, 100, 0)
    chosen = [i for i in test_ids if len(by_id[i]) >= 4][:12]
    files = [f for i in chosen for f in by_id[i][:4]]
    embeddings, index = embed_files(model, files, root, device, workers=2)
    ordered = embeddings[[index[f] for f in files]]
    image = ax.imshow(sim_matrix(ordered).numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("similarity matrix S (4x4 blocks = same identity)")

    # (d) training curves
    ax = axes[1, 1]
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["loss"] for h in history], color=BLUE, label="train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss", color=BLUE)
    twin = ax.twinx()   # the ONE place two scales are justified: different units, one x
    twin.plot(epochs, [h["val_auc"] for h in history], color=GREEN, label="val AUC")
    twin.set_ylabel("val AUC", color=GREEN)
    ax.set_title("training")

    fig.tight_layout()
    path = out_dir / "summary.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
# §8  COMMANDS
# ============================================================================


def cmd_selftest(args):
    """The exact loss vectors. No data, no GPU, ~2 seconds."""
    print("Loss unit vectors (fp32)\n" + "-" * 46)
    ok = True

    def check(name, got, want, tol=1e-5):
        nonlocal ok
        passed = abs(got - want) < tol
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<34} {got:.6f}  (want {want})")

    # Contrastive, m=1.0
    for d, y, want in [(0.5, 1, 0.25), (0.5, 0, 0.25), (1.2, 0, 0.0), (0.0, 1, 0.0)]:
        dd, yy = torch.tensor([d]), torch.tensor([float(y)])
        got = (yy * dd.pow(2) + (1 - yy) * (1.0 - dd).clamp_min(0).pow(2)).item()
        check(f"contrastive d={d} y={y}", got, want)

    # Triplet, alpha=0.2
    for d_ap, d_an, want in [(0.3, 0.9, 0.0), (0.8, 0.9, 0.1)]:
        got = max(0.0, d_ap - d_an + 0.2)
        check(f"triplet d_ap={d_ap} d_an={d_an}", got, want)

    # InfoNCE, tau=1: anchor & positive identical (s=1), negative orthogonal (s=0).
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    loss, _ = infonce_loss(z, torch.tensor([0, 0, 1]), tau=1.0)
    check("infonce tau=1", loss.item(), -math.log(math.e / (math.e + 1)))

    # Geometry
    z1, z2 = l2_normalize(torch.randn(500, 64)), l2_normalize(torch.randn(500, 64))
    s = (z1 * z2).sum(1)
    d = (z1 - z2).norm(dim=1)
    check("d^2 = 2 - 2s", (d.pow(2) - (2 - 2 * s)).abs().max().item(), 0.0)

    # The two NaN gotchas
    zz = l2_normalize(torch.randn(16, 8)).requires_grad_(True)
    dist_matrix(zz).sum().backward()
    check("no NaN grad from sqrt(0) diagonal", float(torch.isfinite(zz.grad).all()), 1.0)

    zz = l2_normalize(torch.randn(8, 8)).requires_grad_(True)
    infonce_loss(zz, torch.arange(4).repeat_interleave(2), supcon=True)[0].backward()
    check("no NaN grad from SupCon -inf*0", float(torch.isfinite(zz.grad).all()), 1.0)

    # Semi-hard band property
    torch.manual_seed(0)
    z = l2_normalize(torch.randn(48, 16))
    labels = torch.arange(12).repeat_interleave(4)
    d = dist_matrix(z)
    a, p, n = mine_triplets(z, labels, "semi-hard", 0.3)
    same = labels[:, None] == labels[None, :]
    violations = 0
    for ai, pi, ni in zip(a.tolist(), p.tolist(), n.tolist()):
        d_ap = d[ai, pi]
        band = (~same[ai]) & (d[ai] > d_ap) & (d[ai] < d_ap + 0.3)
        if band.any() and not (d_ap < d[ai, ni] < d_ap + 0.3):
            violations += 1
    check("semi-hard negatives inside the band", float(violations), 0.0)

    print("-" * 46)
    print("ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def cmd_explain(args):
    """The key ideas, each with a number computed right now."""
    torch.manual_seed(0)
    print(__doc__.split("READING ORDER")[0].strip()[:0] or "", end="")

    print("=" * 72)
    print("1. WHY NORMALISE — cosine and Euclidean become the same question")
    print("=" * 72)
    z1, z2 = l2_normalize(torch.randn(5, 32)), l2_normalize(torch.randn(5, 32))
    for s, d in zip((z1 * z2).sum(1), (z1 - z2).norm(dim=1)):
        print(f"   s = {s:+.4f}   d = {d:.4f}   sqrt(2-2s) = {math.sqrt(2 - 2 * s):.4f}")
    print("   d is strictly decreasing in s, so one threshold serves both.\n")

    print("=" * 72)
    print("2. WHY PK SAMPLING — with K=1 there are NO positives to learn from")
    print("=" * 72)
    labels = torch.arange(32).repeat_interleave(4)      # the default P=32, K=4
    same = labels[:, None] == labels[None, :]
    upper = torch.triu(torch.ones(128, 128, dtype=torch.bool), 1)
    print(f"   P=32, K=4 -> batch 128: {int((same & upper).sum())} positive pairs, "
          f"{int(((~same) & upper).sum())} negative pairs")
    print(f"   imbalance {int(((~same) & upper).sum()) / int((same & upper).sum()):.0f}:1 "
          "-> contrastive MUST subsample negatives or they dominate.\n")

    print("=" * 72)
    print("3. WHY THE POSITIVE STAYS IN THE INFONCE DENOMINATOR")
    print("=" * 72)
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 0, 1])
    with_pos, _ = infonce_loss(z, y, tau=1.0)
    print(f"   perfect match, tau=1 -> loss {with_pos.item():.5f}, NOT 0")
    print(f"   -log(e/(e+1)) = {-math.log(math.e/(math.e+1)):.5f}")
    print("   That floor is what keeps the loss bounded below. Drop the positive")
    print("   from the denominator and the loss can go negative -- unbounded.\n")

    print("=" * 72)
    print("4. WHY TEMPERATURE MATTERS — it decides which negatives get gradient")
    print("=" * 72)
    sims = torch.tensor([0.9, 0.5, 0.1, -0.3])
    print(f"   four negatives at similarities {[round(v, 1) for v in sims.tolist()]}")
    for tau in (0.03, 0.07, 0.5):
        w = torch.softmax(sims / tau, 0)
        print(f"     tau={tau:<5} gradient weight {[round(v, 3) for v in w.tolist()]}")
    print("   Small tau -> almost all gradient on the single hardest negative.\n")

    print("=" * 72)
    print("5. WHY THE MINER MATTERS MORE THAN THE MARGIN")
    print("=" * 72)
    z = l2_normalize(torch.randn(64, 32))
    labels = torch.arange(16).repeat_interleave(4)
    for miner in ("random", "semi-hard", "batch-hard"):
        _, logs = triplet_loss(z, labels, 0.2, miner)
        print(f"   {miner:<11} active fraction {logs['active_fraction']:.3f}  "
              f"d_ap {logs['d_ap']:.3f}  d_an {logs['d_an']:.3f}")
    print("   In the full lab: miner changed LFW by 0.110, margin by 0.005 (22x).\n")

    print("=" * 72)
    print("6. WHY ALIGNMENT ALONE CAN BE FOOLED")
    print("=" * 72)
    labels = np.repeat(np.arange(25), 4)
    collapsed = torch.zeros(100, 16); collapsed[:, 0] = 1.0
    good_centres = l2_normalize(torch.randn(25, 16)).repeat_interleave(4, 0)
    good = l2_normalize(good_centres + 0.1 * torch.randn(100, 16))
    for name, emb in (("collapsed", collapsed), ("healthy", good)):
        m = alignment_uniformity(emb, labels)
        print(f"   {name:<10} alignment {m['alignment']:.4f}  uniformity {m['uniformity']:.4f}")
    print("   Collapse has PERFECT alignment and the worst possible uniformity.")
    print("   You need both numbers, which is why the lab reports both.\n")
    return 0


def cmd_train(args):
    result = train(args)
    if not args.no_plot:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_model(result["dir"] / "model.pt", device)
        by_id = load_identities(Path(args.data))
        _, val_ids, _ = split_identities(by_id, args.train_ids, args.val_ids, args.seed)
        pairs, labels = make_pairs(by_id, val_ids, args.val_pairs, args.seed)
        metrics, scores = evaluate(model, pairs, labels, Path(args.data), device, args.workers)
        path = plot_all(model, scores, labels, metrics, result["history"],
                        result["dir"], Path(args.data), device)
        print(f"figure -> {path}")
    return 0


def cmd_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.ckpt, device)
    saved = checkpoint["args"]

    by_id = load_identities(Path(args.data))
    _, _, test_ids = split_identities(by_id, saved["train_ids"], saved["val_ids"], saved["seed"])
    pairs, labels = make_pairs(by_id, test_ids, args.val_pairs, seed=99)

    print(f"evaluating on {len(pairs):,} pairs from {len(test_ids):,} UNSEEN test identities")
    metrics, _ = evaluate(model, pairs, labels, Path(args.data), device, args.workers)
    for key, value in metrics.items():
        print(f"  {key:<16} {value:.4f}")

    files = [f for i in test_ids[:120] if len(by_id[i]) >= 2 for f in by_id[i][:4]]
    embeddings, index = embed_files(model, files, Path(args.data), device, workers=args.workers)
    ids = [i for i in test_ids[:120] if len(by_id[i]) >= 2 for _ in by_id[i][:4]]
    diagnostics = alignment_uniformity(embeddings[[index[f] for f in files]], np.array(ids))
    for key, value in diagnostics.items():
        print(f"  {key:<16} {value:.4f}")
    return 0


def cmd_compare(args):
    """Train all three losses at an identical budget — the E1 experiment."""
    results = {}
    for name in ("contrastive", "triplet", "infonce"):
        print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
        args.loss, args.margin = name, None
        results[name] = train(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_id = load_identities(Path(args.data))
    _, _, test_ids = split_identities(by_id, args.train_ids, args.val_ids, args.seed)
    pairs, labels = make_pairs(by_id, test_ids, args.val_pairs, seed=99)

    print(f"\n{'=' * 78}\nE1 loss face-off — {len(pairs):,} pairs, UNSEEN test identities")
    print(f"{'=' * 78}")
    print(f"{'loss':<14}{'AUC':>9}{'acc':>9}{'TAR@1e-3':>11}{'EER':>9}{'align':>9}{'unif':>9}")

    for name, result in results.items():
        model, _ = load_model(result["dir"] / "model.pt", device)
        metrics, _ = evaluate(model, pairs, labels, Path(args.data), device, args.workers)
        files = [f for i in test_ids[:120] if len(by_id[i]) >= 2 for f in by_id[i][:4]]
        ids = [i for i in test_ids[:120] if len(by_id[i]) >= 2 for _ in by_id[i][:4]]
        emb, index = embed_files(model, files, Path(args.data), device, workers=args.workers)
        diagnostics = alignment_uniformity(emb[[index[f] for f in files]], np.array(ids))
        print(f"{name:<14}{metrics['auc']:>9.4f}{metrics['accuracy']:>9.4f}"
              f"{metrics['tar@far=0.001']:>11.4f}{metrics['eer']:>9.4f}"
              f"{diagnostics['alignment']:>9.3f}{diagnostics['uniformity']:>9.3f}")

    print("\nExpect InfoNCE >= triplet >= contrastive, with the gap far wider on")
    print("TAR@FAR than on accuracy. In the full 30-epoch lab it was 1.09x on")
    print("accuracy and 5.8x on TAR@FAR=1e-3.")
    return 0


def cmd_verify(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, device)
    transform = build_transform(train=False)
    batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in (args.a, args.b)])
    with torch.no_grad():
        z = l2_normalize(model(batch.to(device)).float().cpu())
    s = float((z[0] * z[1]).sum())
    accept = s > args.threshold
    print(f"\n  A  {args.a}\n  B  {args.b}")
    print(f"  similarity s = {s:.4f}   (d = {math.sqrt(max(0, 2 - 2 * s)):.4f})")
    print(f"  threshold  t = {args.threshold:.4f}")
    print(f"  -> {'ACCEPT (same person)' if accept else 'REJECT (different person)'}\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, quick=False):
        p.add_argument("--data", default="../data/celeba")
        p.add_argument("--out", default="runs")
        p.add_argument("--loss", default="infonce",
                       choices=["contrastive", "triplet", "infonce"])
        p.add_argument("--miner", default="semi-hard",
                       choices=["random", "semi-hard", "batch-hard"])
        p.add_argument("--margin", type=float, default=None, help="m or alpha")
        p.add_argument("--tau", type=float, default=0.07)
        p.add_argument("--dim", type=int, default=128)
        p.add_argument("--no-normalize", action="store_true")
        p.add_argument("--augment", default="basic", choices=["none", "basic", "strong"])
        p.add_argument("--epochs", type=int, default=10 if quick else 20)
        p.add_argument("--batches", type=int, default=300 if quick else 600)
        p.add_argument("--p", type=int, default=32)
        p.add_argument("--k", type=int, default=4)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--train-ids", type=int, default=1500 if quick else 3000)
        p.add_argument("--val-ids", type=int, default=200)
        p.add_argument("--val-pairs", type=int, default=1000)
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--no-plot", action="store_true")

    sub.add_parser("selftest", help="exact loss vectors; no data needed")
    sub.add_parser("explain", help="key ideas with live numbers")

    add_common(sub.add_parser("train", help="train one model"))
    add_common(sub.add_parser("demo", help="quick end-to-end (~1 min, reaches AUC ~0.91)"), quick=True)
    add_common(sub.add_parser("compare", help="all three losses, identical budget (~3 min)"),
               quick=True)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint on TEST identities")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--data", default="../data/celeba")
    p_eval.add_argument("--val-pairs", type=int, default=2000)
    p_eval.add_argument("--workers", type=int, default=6)

    p_verify = sub.add_parser("verify", help="compare two face images")
    p_verify.add_argument("a"); p_verify.add_argument("b")
    p_verify.add_argument("--ckpt", required=True)
    p_verify.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    return {
        "selftest": cmd_selftest, "explain": cmd_explain, "train": cmd_train,
        "demo": cmd_train, "compare": cmd_compare, "eval": cmd_eval, "verify": cmd_verify,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
