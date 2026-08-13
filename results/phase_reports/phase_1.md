# Phase 1 — Minimal pipeline

**Status: gate passed.**

## (a) What was built

| Path | What |
|---|---|
| `src/data/transforms.py` | resize 128 → 112 crop, mean=std=0.5; augmentation levels `none`/`basic`/`strong` |
| `src/data/celeba.py` | `CelebAIdentityDataset` (split-filtered, contiguous labels), `CelebAPairDataset` (deduplicated pair images) |
| `src/models/encoder.py` | ResNet-18 → GAP → Linear(512→d) → BN1d → optional L2 norm, d=128 |
| `src/losses/contrastive.py` | plan §6a formula + the three required per-step logs |
| `src/engine/evaluate.py` | `embed_dataset`, `score_pairs`, `evaluate_pairs` (ROC-AUC) |
| `src/engine/train.py` | single-GPU loop, YAML-driven, resolved config written next to outputs |
| `configs/e0_overfit.yaml`, `configs/baseline_contrastive.yaml` | |
| `tests/test_losses.py`, `tests/test_encoder.py` | contrastive vectors + encoder invariants |

Also added `data/splits/internal_val_pairs.txt` (3,000 pos + 3,000 neg, **val** identities). Phase 0 only produced test pairs, but the gate requires per-epoch val AUC for checkpoint selection — selecting on the test list would bias every number the lab later reports. Verified disjoint from the test list down to the image level.

## (b) Gate evidence

### Gate 1 — E0 passes

```
epoch 199 | loss 0.0603 | mean_d_pos 0.2103 mean_d_neg 1.2677
          | frac_neg_in_margin 0.2969 | val_auc 0.9927
{
  "final": {"auc": 0.992672, "mean_s_pos": 0.98569, "mean_s_neg": 0.05711, "gap": 0.92858},
  "best_val_auc": 0.992672, "best_epoch": 199,
  "final_train_loss": 0.060328,
  "train_images": 219, "train_identities": 10
}
```

**AUC 0.9927 > 0.99 ✓.** Positives sit at cosine 0.986, negatives at 0.057 — a 0.93 gap.

*Honest note on "loss → ≈0":* the loss plateaus at ≈0.06, not 0.00. The floor is real and expected: augmentation level `none` still means **random crop** (plan §3 defines `none` as "crop only"), so the same image presents differently across epochs and cannot be memorized to zero distance. Positives at d=0.21 contribute d²=0.044, essentially the whole remaining loss. The substance of the gate — the pipeline can drive positives together and negatives apart on data it is allowed to memorize — is unambiguous.

### Gate 2 — timing on the full train split

```
train: 158,931 images / 8,000 identities (min 1, median 21, max 35 img/id)
sampler: P=64 K=4 batch=256 620 batches/epoch, 488 ids dropped (<K images)
Encoder(resnet18, d=128, normalize=True, 11.2M params)
epoch 0 | loss 0.4593 | val_auc 0.6889 | 13.1s 12075 img/s
epoch 1 | loss 0.2329 | val_auc 0.7882 | 11.7s 13544 img/s
median_epoch_seconds: 12.43   median_img_per_s: 12767
```

**12.4 s/epoch · 12,767 img/s** at batch 256, bf16, one GPU (48 CPU cores, 8 dataloader workers).

This is far faster than plan §10's "~1–2 h per 30-epoch run" — that estimate assumed A6000-class hardware. A 30-epoch run is **~6–7 minutes**, so the full E1–E8 matrix (~35 runs) projects to roughly **4 hours** on 3 GPUs rather than 2–3 days.

### Gate 3 — resolved config present

`results/e0_overfit/config.yaml` exists, holding defaults merged with the file.

## (c) Surprises & deviations

**1. NaN gradient bug found and fixed — this one mattered.** ⚠️
`test_module_loss_is_differentiable` failed with a non-finite gradient. Cause: `euclidean_distance_matrix` computes `sqrt(2 − 2s)`, and the self-similarity diagonal is `sqrt(0)`, whose derivative is infinite. The losses mask the diagonal, so its upstream gradient is exactly `0` — and `0 × inf = NaN`, which then contaminates the entire backward pass.

Fixed in `geometry._safe_sqrt`, which needs both halves: `clamp_min(eps)` before the sqrt (finite local derivative) **and** `masked_fill` of the exact zeros afterwards (restores the value 0.0 and blocks gradient flow). Clamping alone leaves d=1e-6 on the diagonal; masking alone still evaluates `sqrt'(0)=inf` during backward. Three regression tests added, including the coincident-embedding case that a duplicated image would produce. Confirmed clean across 200 E0 epochs and 2 full-split epochs — no NaN.

**2. "Epoch" redefined to one pass over images, not identities.**
With P=64, K=4 a single pass over the 7,512 usable identities is 117 batches = 29,952 images — under 20% of the split. 30 such epochs would be 5.7 dataset passes and would undertrain badly against the plan's 0.85–0.94 LFW expectation. `batches_per_epoch` now defaults to `len(train_set) // (P·K)` = **620 batches**, so an epoch covers the full split. This also reconciles the plan's own runtime estimate, which only makes sense at full-split epochs. E0 overrides it explicitly and is unaffected.

**3. Losses always run in fp32 even under bf16 autocast.** The embeddings come out of autocast and are cast up before the loss. bf16 carries ~3 decimal digits, which is not enough for the N×N similarity matrix or (in Phase 2) InfoNCE's log-sum-exp.

**4. Contrastive pair balancing is not optional.** A P=64,K=4 batch holds 384 positive pairs and 31,872 negative ones. Left as-is the negative term dominates the gradient ~83:1, so negatives are subsampled to the positive count (plan §4's "balanced"). Tested.

## Learning checkpoint (for Ali)

In E0, sketch what the positive/negative distance histograms look like at loss ≈ 0. Why is overfitting the *goal*, and which bugs does it rule out (labels, sampler, loss sign, optimizer wiring)?

---

Proceeding to Phase 2 without stopping, per your instruction.
