# Siamese Network Training Lab — Implementation Plan
**Contrastive vs. Triplet vs. InfoNCE for Face Verification**

Purpose: a learning-first reimplementation of everything in the reference poster ("Training a Siamese Network — Can we use InfoNCE?"). Every panel of the poster becomes runnable code, an experiment, and a figure, so the owner (Ali) can *see* each claim hold or fail, not just read it.

Audience: an implementing LLM agent (e.g., Claude Code) running on Ali's Linux server. This document is a complete spec — implement phases in order, stop at each acceptance gate, and record results. Where anything is ambiguous, use the stated defaults.

---

## 1. Learning goals (what the finished lab must demonstrate)

Each maps to a poster panel:

| # | Poster panel | Question the lab answers empirically |
|---|---|---|
| 1 | Training objective | Why L2-normalize embeddings? What breaks without it? |
| 2 | Data formation | How are positive/negative pairs and PK batches built? |
| 3 | Forward pass | Shared-weight encoder; cosine vs. Euclidean (and their exact relationship). |
| 4a | Contrastive loss | Effect of margin `m` on the embedding geometry. |
| 4b | Triplet loss | Effect of margin `α` and, more importantly, of negative mining. |
| 4c | InfoNCE | Effect of temperature `τ` and number of negatives (batch size). |
| 5 | Training loop | Full pipeline incl. the batch similarity matrix, visualized live. |
| 6 | Verification | Thresholding cosine similarity → Accept/Reject; ROC, TAR@FAR, EER. |
| 7 | When InfoNCE? | Does InfoNCE beat pairwise losses? Does pre-train→fine-tune help? |
| 8 | Practical tips | Augmentation, balancing, hard negatives, normalization — each ablated. |

---

## 2. Hardware & environment

- **GPUs:** 3× NVIDIA A6000-class (48 GB each; if these are RTX PRO 6000 Blackwell 96 GB, nothing changes except the max batch sizes in E5 can double).
- **Two usage modes:**
  - **Sweep mode (default):** 3 independent single-GPU runs in parallel — one config per GPU. Most experiments run this way.
  - **DDP mode:** one `torchrun --nproc_per_node=3` job for the large-batch InfoNCE experiment (E5), with cross-GPU negative gathering.
- **Stack:** Python 3.11, PyTorch ≥ 2.3 (CUDA), torchvision, scikit-learn, umap-learn, matplotlib, pyyaml, tqdm; wandb optional (off by default, everything also logged to disk).
- Mixed precision bf16 on by default. Seed everything (`torch`, `numpy`, `random`); `cudnn.benchmark=True` for speed, full determinism only inside unit tests.

---

## 3. Data

**Primary training set (default): CelebA** (aligned+cropped; 202,599 images, 10,177 identities), loaded via `torchvision.datasets.CelebA` with `target_type="identity"`.
- Identity-disjoint splits (write once to `data/splits/*.txt`, commit them, assert disjointness at every startup):
  - train: 8,000 identities · val: 1,000 identities · test: remaining ~1,177 identities.
- If the torchvision Google-Drive download hits quota limits, README documents manual download (Kaggle mirror) into `data/celeba/`.

**External verification benchmark: LFW** via `torchvision.datasets.LFWPairs` (test split; 6,000 pairs; standard 10-fold protocol). Used only for evaluation, never training.

**Optional scale-up:** CASIA-WebFace (~0.49 M images / 10.5 K ids) if a local copy exists — the loader just takes a root folder with one subfolder per identity. Do not attempt to download it (redistribution is restricted).

**Debug set:** the first 10 CelebA train identities (used by the E0 overfit sanity run).

**Preprocessing:** resize 128 → random crop 112 (train) / center crop 112 (eval); normalize with mean=std=0.5. Augmentation levels (matches Tips panel):
- `none`: crop only
- `basic`: + horizontal flip
- `strong`: + color jitter 0.4, random grayscale p=0.2, Gaussian blur p=0.5, random erasing p=0.25

Expectation-setting (important for learning calibration): a ResNet-18 trained on CelebA with these losses will land roughly **0.85–0.94 LFW accuracy**, not the 99 %+ of ArcFace-style systems trained on MS1M. This lab is about *relative comparisons between losses and hyperparameters*, not SOTA.

---

## 4. Batch construction: PK sampling (Panel 2)

One sampler serves all three losses. Each batch = **P identities × K images** (default P=64, K=4 → N=256).

- **Contrastive pairs:** within the batch, form all same-identity pairs (y=1) and an equal number of random different-identity pairs (y=0) — balanced, per the Tips panel.
- **Triplets:** every image is an anchor; positive and negative chosen by the active miner (Sec. 6b).
- **InfoNCE:** for each anchor, one in-batch positive (same identity, random); the denominator runs over **all other batch samples including the positive**, exactly as written on the poster.

`PKSampler` must guarantee exactly P ids × K images per batch (unit-tested), reshuffling identities each epoch.

---

## 5. Model (Panels 1 & 3)

```
Encoder f_θ:  torchvision resnet18 (flag: resnet50)
              → global average pool
              → Linear(512 → d)          # d = 128 default
              → BatchNorm1d(d)
              → L2 normalize  ẑ = z / ||z||₂   (flag: --no-normalize for E6)
```

Both "branches" are literally the same `nn.Module` called twice — shared weights by construction (Panel 3). No ImageNet pretraining by default (flag to enable; keep off so loss comparisons are clean).

Similarity/distance utilities (one module, unit-tested):
- cosine: `s(z1,z2) = ẑ1ᵀẑ2 ∈ [-1, 1]` (higher = more similar)
- Euclidean: `d(z1,z2) = ||ẑ1 − ẑ2||₂ ≥ 0` (lower = more similar)
- Key identity to assert in tests: for normalized embeddings, **d² = 2 − 2s**. This is why the two views on the poster are interchangeable.

---

## 6. Losses (Panel 4) — exact formulas, defaults, and logging

### (a) Pairwise contrastive (Hadsell et al., 2006)
```
L(z1, z2, y) = y · d² + (1 − y) · max(0, m − d)²
```
Default `m = 1.0`. Note for the report: with normalized embeddings d ∈ [0, 2], so m must be ≤ 2 — E2 sweeps this.
Log per step: mean d over positives, mean d over negatives, fraction of negatives inside the margin (i.e., still contributing loss).

### (b) Triplet (Schroff et al., 2015)
```
L(a, p, n) = max(0, d(a,p) − d(a,n) + α)
```
Default `α = 0.2`. Miners (Sec. on E3):
- `random`: random positive + random negative per anchor.
- `semi-hard`: negatives with `d_ap < d_an < d_ap + α` (FaceNet); fall back to the hardest violating negative if none exist.
- `batch-hard` (Hermans et al., 2017): hardest positive (max d_ap) and hardest negative (min d_an) per anchor.
Log per step: **fraction of active (non-zero) triplets** — the single most instructive curve for understanding mining.

### (c) InfoNCE (van den Oord et al., 2018)
```
L_i = −log [ exp(s(z_i, z_{i+}) / τ)  /  Σ_{j ∈ B, j ≠ i} exp(s(z_i, z_j) / τ) ]
```
Default `τ = 0.07`, fixed (learnable-τ available behind a flag, off by default — the poster uses fixed τ).
Implementation: compute the full N×N cosine-similarity matrix `S = Ẑ Ẑᵀ`, mask the diagonal, cross-entropy against the positive index. **Save S periodically as a heatmap ordered by identity — this literally recreates Panel 5, step 4.**
Stretch flag `--supcon`: use *all* same-identity samples as positives (Khosla et al., 2020) — natural extension since we have labels.

**DDP global negatives (needed for E5):** gather normalized embeddings across ranks with a gradient-preserving all-gather (`torch.distributed.nn.functional.all_gather`), so the negative pool = the *global* batch. Flag `negatives: local | global`.

---

## 7. Training loop (Panel 5)

`sample PK batch → encode with shared f_θ → similarity matrix → loss (contrastive | triplet | infonce) → backprop → repeat`

Defaults: AdamW, lr 3e-4, weight decay 1e-4, cosine schedule with 5-epoch linear warmup, 30 epochs, batch 256/GPU, bf16 AMP, grad clip 5.0. Evaluate on the internal val pairs every epoch; checkpoint best-by-val-AUC and last. Every run writes `results/<exp_name>/{config.yaml, metrics.json, curves.csv, figures/, ckpts/}` plus an auto-generated `report.md`.

---

## 8. Evaluation suite (Panel 6 + Tips "monitor" line)

1. **Internal verification** (test identities, never seen in training): a fixed, seeded list of 6,000 positive + 6,000 negative pairs committed to `data/splits/`. Score = cosine similarity. Metrics:
   - ROC-AUC; **TAR @ FAR ∈ {1e-3, 1e-2}**; **EER**; best-threshold accuracy.
2. **LFW protocol:** 10-fold; per fold, fit the threshold on the other 9 folds, test on the held-out fold; report **mean ± std accuracy**.
3. **Retrieval** (test ids): 1 gallery image per identity, rest as queries → **Recall@{1, 5, 10}** (mAP optional).
4. **Representation diagnostics** (Wang & Isola, 2020) — explains *why* InfoNCE behaves as it does:
   - alignment = `E_pos ||ẑ_i − ẑ_j||²`  ·  uniformity = `log E_{i,j} exp(−2 ||ẑ_i − ẑ_j||²)`
5. **Panel-6 demo script** `scripts/verify_pair.py img1 img2 --ckpt ...` → prints similarity, threshold, **Accept / Reject**, and saves a side-by-side figure. This is the "face verification example" panel made tangible.

---

## 9. Visualization suite (auto-generated per run; PNG + PDF)

| ID | Figure | Poster panel it brings to life |
|---|---|---|
| V1 | Positive vs. negative cosine-similarity histograms + chosen threshold line | Panel 6 |
| V2 | ROC curves (log-scale FPR axis), one line per loss/config | Panel 6 / Tips |
| V3 | Batch similarity heatmap `S`, rows/cols ordered by identity, snapshots across training | Panel 5.4 |
| V4 | UMAP of embeddings for 30 *unseen* identities, colored by identity | Panel 1 ("close / far apart") |
| V5 | Training curves: loss; active-triplet %; gap = mean s_pos − mean s_neg | Panels 4–5 |
| V6 | Alignment–uniformity scatter across checkpoints and losses | Panel 7 (why InfoNCE transfers) |
| V7 | Sweep plots: metric vs. swept parameter (τ, m, α, batch, d, aug) | Panels 4, 7, 8 |

Top-level `scripts/compare.py` aggregates `metrics.json` across experiments into one comparison table + the cross-experiment versions of V2/V6/V7.

---

## 10. Experiment matrix (the core of the learning)

Each row: hypothesis from the poster → config delta → figure → what Ali should expect to see (so he can check his own predictions before looking — write the prediction down first).

| ID | Question (panel) | Config delta | Key figure | Expected outcome |
|---|---|---|---|---|
| **E0** | Does the pipeline work at all? | 10-id debug set, contrastive, 200 epochs | V5 | Loss → ~0; internal AUC on those ids > 0.99. Gate for Phase 1. |
| **E1** | Loss face-off (4, 7, orange caveat) | contrastive vs. triplet(semi-hard) vs. InfoNCE, identical budget | V1, V2, V6 + table | InfoNCE ≥ triplet ≥ contrastive on LFW/Recall@K; gaps shrink for pure pairwise verification — the poster's caveat. |
| **E2** | Contrastive margin (4a) | m ∈ {0.25, 0.5, 1.0, 1.5} | V1 per m, V7 | Small m → poor separation (negatives stop being pushed); histograms visibly shift. |
| **E3** | Triplet margin × mining (4b, Tips) | α ∈ {0.1, 0.2, 0.4} × miner ∈ {random, semi-hard, batch-hard} | V5 (active-%), V7 | Mining matters more than α. Random: active-% collapses early (wasted triplets); semi-hard/batch-hard keep learning. |
| **E4** | Temperature (4c) | τ ∈ {0.03, 0.05, 0.07, 0.1, 0.2, 0.5} | V7, V6 | U-shaped accuracy, best ≈ 0.05–0.1; uniformity rises as τ falls. |
| **E5** | Negatives / batch size (7) | InfoNCE, per-GPU batch {64, 128, 256} local; {768} global via 3-GPU DDP (+1536 if 96 GB cards) | V7 (acc vs. log batch) | Accuracy grows with negatives, with diminishing returns — the poster's "larger mini-batches" claim, plus a local-vs-global DDP comparison. |
| **E6** | Normalization (1, Tips) | L2-norm on vs. off (fixed InfoNCE) | V1, V5 + embedding-norm curve | Off: similarity scale drifts, unstable training, worse ROC. |
| **E7** | Augmentation (Tips) | none / basic / strong (fixed InfoNCE) | V7 | Strong wins, largest effect for InfoNCE. |
| **E8** | Embedding dim (1) | d ∈ {64, 128, 256, 512} | V7 | Mild effect; plateau by 128–256. |
| **E9** *(stretch)* | Pre-train then fine-tune (7) | InfoNCE 30 ep → contrastive fine-tune 10 ep, vs. contrastive-from-scratch 40 ep | V2, table | Pre-train→fine-tune ≥ from-scratch — the "especially useful before fine-tuning" claim. |

**Scheduling on 3 GPUs:** every row except E5-global is a single-GPU run — keep all 3 GPUs busy with `scripts/sweep.py`, which reads a list of configs and assigns them round-robin to free GPUs. Measure wall-clock in Phase 1; expect ~1–2 h per 30-epoch ResNet-18 run at 112 px, so the full matrix is roughly 2–3 days of unattended compute.

---

## 11. Phased roadmap with acceptance gates

The agent stops at each gate, writes `results/phase_reports/phase_N.md` (what was built, gate evidence, surprises), and only then continues.

- **Phase 0 — Environment & data.** Repo scaffold, deps pinned, CelebA + LFW downloaded, split files generated. *Gate:* `pytest` green on sampler/geometry tests; split-disjointness assertion passes.
- **Phase 1 — Minimal pipeline.** Single-GPU contrastive training end-to-end + internal eval. *Gate:* E0 passes its criteria; wall-clock per epoch recorded.
- **Phase 2 — All losses & miners.** Contrastive, triplet (3 miners), InfoNCE (+ SupCon flag). *Gate:* all unit-test vectors in Sec. 12 pass exactly.
- **Phase 3 — Eval + viz suites.** LFW protocol, TAR@FAR/EER, Recall@K, alignment/uniformity, V1–V6, auto `report.md`. *Gate:* one baseline run produces a complete report with every figure.
- **Phase 4 — DDP + global negatives.** *Gate:* on one fixed batch, single-GPU loss and 3-GPU global-gather loss agree within 1e-5; throughput (img/s) logged for both modes.
- **Phase 5 — Run the matrix.** Execute E1–E8 via the sweep runner; `compare.py` builds the cross-experiment report. *Gate:* comparison table + V7 sweep figures exist for every experiment.
- **Phase 6 — Findings.** `FINDINGS.md`: one section per poster claim, each answered with a figure and a number from *this* lab; list of deviations from expectations; extension ideas (SupCon, ArcFace head for contrast, MoCo-style memory queue, cross-dataset test on CASIA).

---

## 12. Unit tests (exact vectors — implement verbatim)

- **Geometry:** for random L2-normalized pairs, assert `d² ≈ 2 − 2s` (atol 1e-5).
- **Contrastive** (m = 1.0): (d=0.5, y=1) → 0.25 · (d=0.5, y=0) → 0.25 · (d=1.2, y=0) → 0 · (d=0, y=1) → 0.
- **Triplet** (α = 0.2): (d_ap=0.3, d_an=0.9) → 0 · (d_ap=0.8, d_an=0.9) → 0.1.
- **InfoNCE** (τ = 1): one anchor, s₊ = 1 and a single negative s₋ = 0 → L = −log(e / (e + 1)) ≈ **0.31326**.
- **Semi-hard miner:** on a crafted batch, every returned negative satisfies `d_ap < d_an < d_ap + α`.
- **Metrics:** perfectly separated scores → EER = 0, AUC = 1; shuffled labels → AUC ≈ 0.5 ± 0.05.
- **PKSampler:** every batch contains exactly P identities × K images.
- **DDP:** gradient-preserving all-gather — loss and encoder grads match the single-process computation on a fixed batch.

---

## 13. Repository layout

```
siamese-lab/
├── configs/                      # one YAML per experiment (E0…E9 committed)
├── data/splits/                  # committed identity splits + eval pair lists
├── src/
│   ├── data/      celeba.py · lfw.py · folder_dataset.py · pk_sampler.py · transforms.py
│   ├── models/    encoder.py                     # backbone + head + optional L2 norm
│   ├── losses/    contrastive.py · triplet.py · infonce.py · miners.py · geometry.py
│   ├── engine/    train.py (single + DDP) · gather.py · evaluate.py
│   ├── metrics/   verification.py (ROC, TAR@FAR, EER, LFW 10-fold) · retrieval.py · align_uniform.py
│   └── viz/       histograms.py · roc.py · heatmap.py · embedding_map.py · curves.py · sweep_plots.py
├── scripts/       run.sh <config> [gpu] · sweep.py · compare.py · verify_pair.py
├── tests/         test_geometry.py · test_losses.py · test_miners.py · test_metrics.py · test_sampler.py · test_ddp_gather.py
├── results/       <exp_name>/{config.yaml, metrics.json, curves.csv, figures/, ckpts/, report.md}
└── README.md      # setup, dataset fallback instructions, how to run each phase
```

---

## 14. Handoff instructions for the implementing agent

1. Read this file fully before writing code; implement phases strictly in order and respect every gate.
2. Style: typed Python, modules ≲ 200 lines, each loss's docstring states the exact formula it implements; plain PyTorch only (no Lightning/fancy frameworks).
3. Every run is driven by a YAML in `configs/`; the resolved config is copied next to its outputs. No hyperparameter may live only in code.
4. Assert train/val/test identity disjointness at startup of every run. Never let LFW touch training.
5. Figures always saved as both PNG and PDF; every number in a report must also exist in `metrics.json`.
6. If CelebA auto-download fails, stop and print the manual-download instructions rather than silently substituting data.
7. When a result contradicts the "expected outcome" column, do not tune it away — flag it in the phase report; contradictions are learning material here.

---

## 15. References

- Hadsell, Chopra, LeCun — *Dimensionality Reduction by Learning an Invariant Mapping*, CVPR 2006 (contrastive loss)
- Schroff, Kalenichenko, Philbin — *FaceNet*, CVPR 2015 (triplet loss, semi-hard mining)
- van den Oord, Li, Vinyals — *Representation Learning with Contrastive Predictive Coding*, 2018 (InfoNCE)
- Hermans, Beyer, Leibe — *In Defense of the Triplet Loss*, 2017 (batch-hard, PK sampling)
- Chen et al. — *SimCLR*, ICML 2020 (temperature, batch size, augmentation, cross-GPU negatives)
- Khosla et al. — *Supervised Contrastive Learning*, NeurIPS 2020 (SupCon extension)
- Wang & Isola — *Alignment and Uniformity on the Hypersphere*, ICML 2020 (diagnostics in Sec. 8.4)
- Musgrave, Belongie, Lim — *A Metric Learning Reality Check*, ECCV 2020 (why fair budgets and proper eval protocols matter — the discipline behind E1)
