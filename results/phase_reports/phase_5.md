# Phase 5 — Experiment matrix

**Status: gate passed.** 35 runs, 0 failures, 0 retries.

## (a) Execution

| | |
|---|---|
| single-GPU runs | 34, round-robin on 3 GPUs via `scripts/sweep.py` |
| DDP run | 1 (`e5_infonce_global768`, `torchrun --nproc_per_node=3`) |
| wall-clock | **87.2 min** for the sweep + ~4 min for the DDP job |
| failures / retries | **0 / 0** |
| per run | ~6.4 min (30 epochs, 12.9 s/epoch incl. per-epoch val eval) |

Every run has a complete folder: `config.yaml`, `metrics.json`, `curves.csv`, `report.md`, `ckpts/`, and V1–V6 as PNG + PDF. `scripts/compare.py` produced `results/comparison/` with the master table and cross-experiment V2/V6/V7.

Plan §10 estimated 2–3 days for this matrix on A6000-class hardware; the Blackwell cards did it in **under 1.5 hours**.

## (b) Gate evidence

1. ✅ All 35 runs have complete results folders.
2. ✅ `results/comparison/` contains `master_table.md` / `.csv`, `v2_roc_e1`, `v6_align_uniform`, and `v7_e2` … `v7_e8`.
3. ✅ Headline numbers below, no interpretation (that is Phase 6).

## (c) Headline numbers per experiment

### E1 — loss face-off (identical budget)

| run | LFW | AUC | TAR@1e-3 | R@1 |
|---|---|---|---|---|
| e1_contrastive | 0.8348 ± 0.0201 | 0.9364 | 0.0780 | 0.1100 |
| e1_triplet_semihard | 0.8980 ± 0.0100 | 0.9662 | 0.3125 | 0.3786 |
| **e1_infonce** | **0.9072 ± 0.0109** | **0.9785** | **0.4495** | **0.4884** |

Figures: `results/comparison/v2_roc_e1.png`, `results/comparison/v6_align_uniform.png`

### E2 — contrastive margin m

| m | LFW | TAR@1e-3 | alignment | uniformity |
|---|---|---|---|---|
| 0.25 | 0.7798 ± 0.0165 | 0.0213 | 0.0075 | −0.0889 |
| 0.5 | 0.8033 ± 0.0189 | 0.0603 | 0.0277 | −0.3783 |
| 1.0 | 0.8278 ± 0.0164 | 0.0657 | 0.1018 | −1.2348 |
| **1.5** | **0.8453 ± 0.0171** | 0.0777 | 0.2159 | −2.0025 |

Figure: `results/comparison/v7_e2.png`

### E3 — triplet margin α × miner

| α | random | semi-hard | batch-hard |
|---|---|---|---|
| 0.1 | 0.8767 ± 0.0141 | **0.8980 ± 0.0146** | 0.8102 ± 0.0141 |
| 0.2 | 0.8788 ± 0.0151 | **0.8980 ± 0.0151** | 0.7877 ± 0.0165 |
| 0.4 | 0.8757 ± 0.0121 | 0.8930 ± 0.0129 | 0.8165 ± 0.0147 |

Figure: `results/comparison/v7_e3.png`

### E4 — InfoNCE temperature τ

| τ | LFW | TAR@1e-3 | alignment | uniformity |
|---|---|---|---|---|
| 0.03 | 0.9032 ± 0.0126 | 0.4050 | 0.2358 | −1.4487 |
| **0.05** | 0.9037 ± 0.0124 | **0.4702** | 0.3936 | −2.2234 |
| 0.07 | 0.9035 ± 0.0141 | 0.4542 | 0.5461 | −2.9132 |
| 0.1 | 0.9037 ± 0.0147 | 0.4028 | 0.6890 | −3.4858 |
| 0.2 | 0.8970 ± 0.0145 | 0.3232 | 0.5901 | −3.3976 |
| 0.5 | 0.8607 ± 0.0184 | 0.1225 | 0.4052 | −2.7228 |

Figure: `results/comparison/v7_e4.png`

### E5 — negatives / batch size

| negatives per anchor | LFW | TAR@1e-3 | R@1 | alignment | uniformity |
|---|---|---|---|---|---|
| 64 (local) | **0.9078 ± 0.0101** | 0.3177 | 0.4615 | 0.3231 | −2.2147 |
| 128 (local) | 0.9065 ± 0.0157 | 0.4150 | 0.4807 | 0.4308 | −2.5817 |
| 256 (local) | 0.9035 ± 0.0169 | 0.4503 | 0.4877 | 0.5468 | −2.9035 |
| **768 (global, 3-GPU DDP)** | 0.8852 ± 0.0130 | **0.4582** | 0.4393 | 0.7347 | **−3.2202** |

Figure: `results/comparison/v7_e5.png`. The 768 row used the Phase-4-validated gradient-preserving gather (`negatives: global`, `world_size=3`, SyncBN, global batch 768, 206 steps/epoch).

### E6 — L2 normalization on/off

| run | LFW | TAR@1e-3 | alignment | uniformity |
|---|---|---|---|---|
| e6_norm_on | 0.9067 ± 0.0138 | 0.4130 | 0.5459 | −2.9141 |
| e6_norm_off | 0.9083 ± 0.0166 | 0.4658 | 0.5434 | −2.8958 |

⚠️ **This experiment did not test what it was supposed to** — see (d)1.

### E7 — augmentation

| level | LFW | TAR@1e-3 | R@1 |
|---|---|---|---|
| none | **0.9038 ± 0.0137** | 0.4362 | 0.4777 |
| basic | 0.9027 ± 0.0145 | 0.4002 | 0.4926 |
| strong | 0.8877 ± 0.0118 | 0.3507 | 0.4524 |

### E8 — embedding dimension d

| d | LFW | TAR@1e-3 | R@1 |
|---|---|---|---|
| 64 | 0.9098 ± 0.0148 | 0.4423 | 0.4855 |
| 128 | 0.9080 ± 0.0159 | 0.4488 | 0.4981 |
| **256** | **0.9112 ± 0.0119** | **0.5175** | 0.5004 |
| 512 | 0.9002 ± 0.0142 | 0.3923 | 0.4974 |

Figure: `results/comparison/v7_e8.png`

## (d) Surprises & deviations — flagged, not tuned away

### 1. ⚠️ E6 is a no-op. The normalization ablation cannot work as built.

`model.normalize=False` never reaches any loss, because `cosine_similarity_matrix` L2-normalizes its own inputs — and `l2_normalize(l2_normalize(z)) == l2_normalize(z)`. Verified directly:

```
InfoNCE(raw, scale 7.3x)  = 5.0444622040
InfoNCE(normalised)       = 5.0444622040
|diff|                    = 0.000e+00
```

So `e6_norm_on` and `e6_norm_off` are the *same experiment run twice*, and the poster's normalization claim is **untested** by this lab as built. Testing it properly needs losses that consume raw dot products and un-normalized Euclidean distances — a code change, not a config change, and therefore out of scope for Phase 5.

**Silver lining: E6 accidentally measured the run-to-run noise floor.** Two identical configurations differing only in nondeterminism landed at LFW 0.9067 vs 0.9083 — a spread of **0.0016**. That number is the yardstick for every other comparison in this matrix, and I use it throughout Phase 6.

### 2. ⚠️ E5 contradicts the poster on LFW but confirms it on TAR@FAR.

LFW accuracy *decreases* with more negatives (0.9078 → 0.8852), while TAR@FAR=1e-3 *increases monotonically* (0.3177 → 0.4150 → 0.4503 → 0.4582) and uniformity improves monotonically (−2.21 → −2.58 → −2.90 → −3.22). Alignment worsens in lockstep (0.32 → 0.73). Interpretation deferred to Phase 6.

Budget caveat, stated for honesty: all four rows see the same number of *images* per epoch, so larger batches take proportionally **fewer optimizer steps** (batch 64 → 2,483 steps/epoch; batch 768 → 206). That is the standard batch-size trade-off, but it means E5 is "equal image budget", not "equal step count".

### 3. ⚠️ E3 contradicts the plan: batch-hard is the *worst* miner, not among the best.

Plan §10 predicted "semi-hard/batch-hard keep learning" against random collapsing. Actual: semi-hard (0.898) > random (0.877) > **batch-hard (0.788–0.817)**, with batch-hard 6–11 points *below* random at every α. Its uniformity is also far worse (−0.52 to −0.85 vs −2.5 to −3.3), which points at partial embedding collapse.

The headline prediction — *mining matters more than α* — is confirmed emphatically: the miner spread is ~0.11 LFW while the α spread within a miner is ~0.005, a **20× ratio**.

### 4. ⚠️ E7 contradicts the plan: strong augmentation is the worst, not the best.

Predicted "strong wins, largest effect for InfoNCE". Actual: none (0.9038) ≈ basic (0.9027) > strong (0.8877). The none-vs-basic gap (0.0011) is *below* the 0.0016 noise floor, so those two are indistinguishable; strong is genuinely worse.

### 5. E2 found no optimum inside the swept range.

LFW rises monotonically to m=1.5 with no turnover. Since normalized embeddings cap d at 2, the sweep simply does not bracket the maximum. Plan §10's "small m → poor separation" is confirmed; "which m is best" is not answered.

### 6. E8 is mostly within noise.

d ∈ {64, 128, 256} spans 0.9080–0.9112 (0.0032, i.e. 2× the noise floor); d=512 at 0.9002 is ~6× the noise floor below the best. The plan's "mild effect; plateau by 128–256" is confirmed, with a hint that 512 over-parameterizes at this data scale.

## Learning checkpoint (Ali — do this BEFORE reading Phase 6)

Write down your predicted outcome for each of E1–E8 next to plan §10's "expected outcome" column. Phase 6 scores those predictions against the curves.

---

Proceeding to Phase 6.
