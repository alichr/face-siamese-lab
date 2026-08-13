# Phase Briefs — Siamese Network Training Lab

Companion to `HANDOFF_PROTOCOL.md` and `siamese_infonce_implementation_plan.md` (the plan). Each brief is self-contained: an agent session given the plan + poster image + protocol + one brief has everything it needs. **Execute only the named phase. Stop at its gate.**

Every phase ends the same way: write `results/phase_reports/phase_N.md` containing (a) what was built, (b) raw gate evidence — pasted test output, printed numbers, figure paths, (c) surprises or deviations from the plan, then **STOP and wait for approval.**

---

## Phase 0 — Environment & data

**Scope:** repo scaffold, dependencies, datasets, splits, and the two foundational modules (sampler, geometry). Nothing else — no model, no losses, no training code.

**Read:** plan §2 (environment), §3 (data), §4 (sampler), §13 (layout). Poster panels 1–2.

**Build**
- Repo scaffold exactly per plan §13; `pyproject.toml` or `requirements.txt` with pinned versions; `README.md` covering setup and the CelebA manual-download fallback.
- Download CelebA (aligned) and LFW pairs via torchvision. If CelebA auto-download fails, stop and print manual instructions — never substitute other data.
- Generate and commit identity-disjoint splits (train 8,000 / val 1,000 / test ~1,177 identities) to `data/splits/`, plus the fixed seeded internal eval pair lists (6,000 pos + 6,000 neg from test identities).
- `src/data/pk_sampler.py` (P identities × K images per batch) and `src/losses/geometry.py` (cosine sim, Euclidean distance, L2 normalize).
- `tests/test_sampler.py`, `tests/test_geometry.py`.

**Gate (all must pass)**
1. `pytest tests/test_sampler.py tests/test_geometry.py` green. Geometry test asserts `d² = 2 − 2s` for random normalized pairs (atol 1e-5). Sampler test asserts every batch is exactly P ids × K images.
2. A startup assertion proves train/val/test identity sets are pairwise disjoint; print the counts per split.
3. Report dataset stats: images and identities per split; min/median images per identity in train.

**Learning checkpoint (Ali, before approving):** why must splits be identity-disjoint rather than image-disjoint — what exactly would leak into verification metrics otherwise? Why does `d² = 2 − 2s` make the poster's "cosine similarity" and "Euclidean distance" boxes two views of the same thing?

**STOP.**

---

## Phase 1 — Minimal pipeline

**Scope:** the smallest end-to-end system: encoder + contrastive loss + single-GPU training + internal AUC eval + YAML configs. Only the contrastive loss exists after this phase.

**Read:** plan §5 (model), §6a (contrastive), §7 (loop defaults). Poster panels 3, 4a, 5.

**Build**
- `src/models/encoder.py` per plan §5 (ResNet-18 → GAP → Linear(512→d) → BN1d → optional L2 norm; d=128).
- `src/losses/contrastive.py` (formula in plan §6a) with per-step logging: mean d over positives, mean d over negatives, fraction of negatives inside the margin.
- `src/engine/train.py` single-GPU path with plan §7 defaults; YAML-driven (`configs/`), resolved config copied next to outputs.
- Minimal eval: ROC-AUC on the internal val pair list each epoch; checkpoint best-by-val-AUC and last.
- `configs/e0_overfit.yaml` (10 debug identities, contrastive, 200 epochs) and `configs/baseline_contrastive.yaml`.

**Gate**
1. **E0 passes:** training loss → ≈0 and internal AUC on the 10 debug identities > 0.99. Paste the final-epoch numbers.
2. One timing epoch on the full train split: report wall-clock/epoch and img/s at batch 256, bf16, on one GPU. (This calibrates the Phase 5 schedule.)
3. Resolved config is present in the E0 results folder.

**Learning checkpoint:** in E0, sketch what the positive/negative *distance* histograms must look like once loss ≈ 0. Why is overfitting the *goal* here, and what specific bugs does this sanity test rule out (labels, sampler, loss sign, optimizer wiring)?

**STOP.**

---

## Phase 2 — All losses & miners

**Scope:** triplet loss with three miners, InfoNCE (with SupCon flag, default off), and the exact unit-test vectors. No DDP, no new eval code.

**Read:** plan §6b–c. Poster panels 4b–c.

**Build**
- `src/losses/triplet.py` (`L = max(0, d_ap − d_an + α)`, default α=0.2) logging the fraction of active (non-zero) triplets per step.
- `src/losses/miners.py`: `random`, `semi-hard` (`d_ap < d_an < d_ap + α`, fallback to hardest violating negative), `batch-hard` (hardest positive, hardest negative per anchor).
- `src/losses/infonce.py`: full N×N cosine matrix, diagonal masked, denominator over all other samples *including* the positive, default τ=0.07 fixed; `--supcon` flag; periodic identity-ordered heatmap dump of S (recreates poster panel 5.4).
- `tests/test_losses.py`, `tests/test_miners.py`.

**Gate — every vector must pass exactly (fp32)**
1. Contrastive (m=1.0): (d=0.5, y=1) → **0.25**; (d=0.5, y=0) → **0.25**; (d=1.2, y=0) → **0**; (d=0, y=1) → **0**.
2. Triplet (α=0.2): (d_ap=0.3, d_an=0.9) → **0**; (d_ap=0.8, d_an=0.9) → **0.1**.
3. InfoNCE (τ=1): one anchor with positive similarity 1 and a single negative with similarity 0 → **−log(e/(e+1)) ≈ 0.31326** (atol 1e-5).
4. Semi-hard miner on a crafted batch returns only negatives satisfying `d_ap < d_an < d_ap + α`.
5. Smoke test: 2 epochs of each loss on the debug set completes without error; loss decreases.

**Learning checkpoint:** hand-derive the 0.31326 value. Why does including the positive in the InfoNCE denominator bound the loss above zero without changing what minimizes it? For batch-hard mining, why does PK sampling (K > 1) make "hardest positive" meaningful at all?

**STOP.**

---

## Phase 3 — Evaluation + visualization suites

**Scope:** the full measurement apparatus and figure generation. One real baseline run to prove it end-to-end. No DDP yet.

**Read:** plan §8 (evaluation), §9 (figures V1–V6). Poster panel 6 + Tips "monitor" line.

**Build**
- `src/metrics/verification.py`: ROC-AUC, TAR@FAR ∈ {1e-3, 1e-2}, EER, best-threshold accuracy; LFW 10-fold protocol (threshold fit on 9 folds, tested on the 10th; mean ± std).
- `src/metrics/retrieval.py`: Recall@{1,5,10} on test identities (1 gallery image per id).
- `src/metrics/align_uniform.py`: alignment and uniformity per plan §8.4.
- `src/viz/`: V1 pos/neg similarity histograms + threshold; V2 ROC (log-FPR); V3 batch similarity heatmap snapshots; V4 UMAP of 30 unseen identities; V5 training curves (loss, active-triplet %, mean s_pos − mean s_neg); V6 alignment–uniformity scatter. All PNG + PDF.
- Auto `report.md` per run; `scripts/verify_pair.py img1 img2 --ckpt` printing similarity, threshold, **Accept/Reject** with a side-by-side figure (poster panel 6, made tangible).
- `tests/test_metrics.py`: perfectly separated scores → EER = 0, AUC = 1; shuffled labels → AUC ≈ 0.5 ± 0.05.

**Gate**
1. Metric tests pass.
2. One full 30-epoch baseline (InfoNCE, defaults) completes: its results folder contains config, metrics.json, curves.csv, **all of V1–V6**, and a rendered report.md. Paste the LFW mean ± std and internal TAR@FAR numbers.
3. Plausibility band: baseline LFW accuracy > 0.80 (plan §3 expects 0.85–0.94; below 0.80 means a bug, not a hyperparameter issue — investigate before passing).
4. `verify_pair.py` demo works on one same-identity and one different-identity pair from test ids.

**Learning checkpoint:** why is the LFW threshold fit on 9 folds and tested on the 10th — what bias appears if you pick the best threshold on the test fold itself? Why report TAR@FAR rather than plain accuracy for verification systems? What does the V1 histogram overlap region correspond to in ROC terms?

**STOP.**

---

## Phase 4 — DDP + global negatives ⚠️ CRITICAL GATE

**Scope:** 3-GPU `torchrun` training and gradient-preserving cross-GPU negative gathering for InfoNCE. This phase exists because the failure mode is *silent*: with a broken gather, training still runs and loss still falls, but gradients from other ranks' anchors never reach the local embeddings — you get small-batch InfoNCE while believing you have large-batch InfoNCE, which would invalidate experiment E5.

**Read:** plan §6c (global negatives), §2 (DDP mode). Poster panel 7 ("larger mini-batches → stronger signal").

**Build**
- `src/engine/gather.py`: gradient-preserving all-gather of normalized embeddings (`torch.distributed.nn.functional.all_gather`); InfoNCE flag `negatives: local | global`.
- DDP path in `src/engine/train.py` (`torchrun --nproc_per_node=3`); decide and document BN handling (SyncBN for training runs; see gate note on eval mode).
- `tests/test_ddp_gather.py` implementing the two-stage equivalence test below (launched via torchrun with 3 processes; fp32 throughout, no AMP).

**Gate — two-stage equivalence test, all tolerances binding**

*Stage A — loss module in isolation (deterministic, synthetic embeddings).*
1. `torch.manual_seed(0)`; create `Z = normalize(randn(258, 128))` with pair structure: 129 identities × 2 images; 258 = 86 per rank.
2. Single-process reference: `L_ref = infonce(Z)` with `Z.requires_grad=True`; keep `dL/dZ`.
3. 3-process run: each rank holds its 86-row slice (requires_grad), gathers globally, computes InfoNCE over its **local anchors** against **global negatives**, averages across ranks. Assert `|L_ddp − L_ref| < 1e-5`.
4. **Gradient check (the part that catches the silent bug):** each rank's `dL/dZ_local` must equal the corresponding 86-row slice of the single-process `dL/dZ`, max abs diff < 1e-5. With a detached gather this fails immediately.
5. Recommended negative control: repeat with plain `dist.all_gather` (detached) and show the gradient check *fails* — include the mismatch magnitude in the report as evidence the test has teeth.

*Stage B — end-to-end (real encoder, fixed inputs).*
6. Fixed batch of 258 real images, model in **eval mode** (freezes BN stats and dropout so single- and multi-process forward passes are comparable), fp32. Compare loss (< 1e-5) and the gradients of a few named encoder parameters (relative error < 1e-4) between single-process and 3-process global-gather runs.
7. Sanity: with `negatives: local`, each rank's loss equals the single-process loss computed on that rank's shard alone.
8. Throughput: log img/s for single-GPU vs 3-GPU DDP at the training config.

**Classic bugs to explicitly check against** (each should be impossible given the tests above, but verify in code review): detached `all_gather` killing gradients; diagonal/self-similarity masking using *local* indices after gathering (must offset by `rank × local_size`); positive-index not offset into the global matrix; loss reduction as mean-of-means when ranks would have unequal anchor counts; forgetting embeddings are normalized *before* gathering; BN batch statistics differing between the single- and multi-process forward (hence eval mode in Stage B, SyncBN documented for real training).

**Learning checkpoint:** write out ∂L_i/∂z_j for a *negative* j in InfoNCE and observe it is nonzero — this is exactly why negatives need gradients and why a detached gather silently changes the objective. Then explain in one paragraph what objective you are *actually* optimizing when the gather is detached, and why loss still decreases.

**STOP.**

---

## Phase 5 — Run the experiment matrix

**Scope:** execute E1–E8 from plan §10 and build the cross-experiment comparison. This is mostly compute, not code.

**Read:** plan §10 (full matrix — authoritative for configs), §2 (scheduling).

**Build / run**
- `scripts/sweep.py`: reads a list of config paths, assigns runs round-robin to free GPUs, restarts on the next config when one finishes; writes a live status table.
- Commit `configs/e1_*.yaml … e8_*.yaml` exactly per the plan's matrix. E5's global-negatives row is the one `torchrun` DDP job; everything else is single-GPU.
- Run order: **E1 first** (its three runs are the baselines every other comparison references), then the remaining rows in any order that keeps all 3 GPUs busy.
- `scripts/compare.py`: aggregates every `metrics.json` into one table + cross-experiment V2 (ROC overlay), V6 (alignment–uniformity), and V7 (one sweep plot per experiment).

**Operational rules for a multi-day job**
- Report incrementally: append to the phase report as each experiment row completes (its table row + key figure paths) rather than waiting for everything.
- Any crashed/NaN run: record it, restart once with the same config; if it fails twice, flag and move on — do not silently change hyperparameters.

**Gate**
1. Every E1–E8 run has a complete results folder (config, metrics.json, figures, report.md).
2. `compare.py` output exists: master table + cross-experiment V2/V6/V7 figures.
3. The phase report lists, per experiment, the headline number (LFW mean ± std or the swept-metric curve) with figure paths — no interpretation yet; Phase 6 does that.

**Learning checkpoint (do this BEFORE opening any Phase 5 figure):** for each of E1–E8, write down your predicted outcome next to the plan's "expected outcome" column. Scoring your predictions against the real curves in Phase 6 is where this project pays off.

**STOP.**

---

## Phase 6 — Findings

**Scope:** interpretation and write-up. No new experiments unless a contradiction demands one targeted follow-up run (flag it first).

**Read:** plan §1 (the panel→question table), §10 expected outcomes; Ali's written predictions from Phase 5.

**Build**
- `FINDINGS.md`: one section per row of the plan §1 table. Each section contains: the poster's claim → the experiment(s) that tested it → the figure and the number → a one-word verdict (**supported / refuted / nuanced**) → two or three sentences of explanation.
- A dedicated section scoring Ali's Phase 5 predictions against results.
- A contradictions section: anything that disagreed with the plan's expected outcomes, with the honest evidence (plan §14.7 — never tuned away).
- Extensions section: SupCon run, ArcFace-head comparison, MoCo-style memory queue, cross-dataset CASIA evaluation — each with one sentence on what it would teach.

**Gate**
1. Every poster panel in the plan §1 table has a verdict backed by a figure path and a number from this lab.
2. The E5 section explicitly uses Phase 4's validated global gather (state it — this closes the loop on the critical gate).
3. FINDINGS.md renders cleanly and stands alone: a reader with only the poster and this file understands what was tested and learned.

**Learning checkpoint:** the exit exam. Without looking at the code, explain to a colleague (or write down): when you would pick contrastive vs triplet vs InfoNCE for a new verification problem, what τ and batch size buy you, and which single practical tip from poster panel 8 mattered most in *your* runs.

**Done.** Project complete after Ali's sign-off.
