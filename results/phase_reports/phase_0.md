# Phase 0 — Environment & data

**Status: gate passed.** Awaiting approval before Phase 1.

---

## (a) What was built

| Path | What it is |
|---|---|
| `docs/` | `plan.md`, `PHASE_BRIEFS.md`, `HANDOFF_PROTOCOL.md`, `poster.png` (moved here on request) |
| `README.md` | setup, CelebA manual-download fallback, layout, phase progress |
| `requirements.txt` | pinned deps incl. the CUDA 13.0 torch index |
| `pyproject.toml` | package metadata, pytest config, ruff config |
| `src/losses/geometry.py` | `l2_normalize`, cosine / Euclidean row-wise + matrix forms, `d↔s` conversions |
| `src/data/pk_sampler.py` | `PKSampler` — exactly P identities × K images per batch |
| `src/data/splits.py` | split loading + `assert_identity_disjoint` (runs at every startup) |
| `scripts/make_splits.py` | one-shot generator for splits, eval pairs, stats |
| `tests/test_geometry.py` | 13 tests |
| `tests/test_sampler.py` | 23 tests |
| `tests/test_splits.py` | 10 tests (added — guards gate item 2; see deviations) |
| `data/splits/` | committed splits + 12,000 eval pairs + `stats.md` |

Scaffold matches plan §13. No model, no losses, no training code — per scope.

---

## (b) Gate evidence

### Gate 1 — `pytest` green

```
$ .venv/bin/python -m pytest tests/ -v
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/alichr/projects/face-siamese-lab
configfile: pyproject.toml
collected 46 items

tests/test_geometry.py .............                                     [ 28%]
tests/test_sampler.py .......................                            [ 78%]
tests/test_splits.py ..........                                          [100%]

============================== 46 passed in 0.74s ==============================
```

**Geometry — `d² = 2 − 2s` at atol 1e-5.** Asserted row-wise (`test_d_squared_equals_2_minus_2s_rowwise`), on the full N×N matrix (`..._matrix`), and again in fp32 rather than fp64 (`test_float32_identity_holds_at_tolerance`), since training runs in fp32/bf16. Also asserted: monotone decrease of `d` in `s` (why thresholding either is the same decision), the three anchors s=1→d=0, s=0→d=√2, s=−1→d=2, and that the masked diagonal gives exactly s=1 / d=0 with no NaN out of `sqrt`.

**Sampler — every batch exactly P ids × K images.** `test_every_batch_is_exactly_p_ids_by_k_images` is parameterized over (P,K) ∈ {(4,2),(8,4),(16,4),(10,3),(64,4)} and asserts, for *every* batch: `len(batch) == P*K`, exactly P distinct identities, and per-identity counts all equal K. Repeated against CelebA-like ragged counts in `test_ragged_labels_still_satisfy_the_gate`.

Live check at the real default P=64, K=4:

```
PKSampler(P=64,K=4): 7,512 usable ids, 488 dropped, batch=256, 117 batches/epoch
  first batch: 256 indices, 64 identities, per-id counts all == {4}
```

### Gate 2 — pairwise-disjoint identity sets

```
$ .venv/bin/python scripts/make_splits.py
CelebA: 202,599 images / 10,177 identities
identity splits: train=8,000 val=1,000 test=1,177 (total 10,177, pairwise disjoint)
internal eval pairs: 6,000 positive + 6,000 negative
```

`assert_identity_disjoint` checks all three pairs (train∩val, train∩test, val∩test) and raises a message naming which pair leaked. It runs inside `load_splits`, so every future run trips it at startup. Its failure path is tested, not just its success path (`test_overlapping_splits_raise`, parameterized over all three leak directions).

Stronger check, `test_eval_pairs_never_reference_a_training_identity`: resolves all 24,000 filenames in the eval list back to identities and asserts the used set is a subset of `test` — and that all 6,000 "positive" pairs really are same-identity and all 6,000 "negative" pairs really are cross-identity. Both hold exactly.

Counts: **train 8,000 · val 1,000 · test 1,177** (total 10,177 = every CelebA identity, used exactly once).

### Gate 3 — dataset stats

| split | identities | images | min img/id | median img/id | max img/id |
|---|---|---|---|---|---|
| train | 8,000 | 158,931 | 1 | 21 | 35 |
| val | 1,000 | 20,134 | 1 | 21 | 31 |
| test | 1,177 | 23,534 | 1 | 21 | 35 |

Total: 202,599 images / 10,177 identities. **Train min/median images per identity: 1 / 21.**

Also written to `data/splits/stats.md`.

### Environment

```
python   3.12.3
torch    2.13.0+cu130 | cuda 13.0 | gpus 3
torchvis 0.28.0+cu130
numpy 2.4.4 | sklearn 1.9.0 | umap 0.5.12 | mpl 3.11.1 | pytest 9.1.1
  gpu0: NVIDIA RTX PRO 6000 Blackwell Max-Q  sm_120  95GiB
  gpu1: NVIDIA RTX PRO 6000 Blackwell Max-Q  sm_120  95GiB
  gpu2: NVIDIA RTX PRO 6000 Blackwell Max-Q  sm_120  95GiB
bf16 matmul on cuda: OK
NCCL available: True          # required for the Phase 4 DDP gate
```

Datasets on disk: `data/celeba/` 3.1 GB (202,599 images + all 5 metadata files) · `data/lfw_sklearn/` 755 MB (6,000 pairs, 3,000 pos / 3,000 neg, funneled, official `pairs.txt`).

---

## (c) Surprises & deviations

**1. LFW cannot be downloaded via torchvision — upstream is gone.** ⚠️ *Affects Phase 3.*
`torchvision.datasets.LFWPairs(download=True)` now raises `ValueError: LFW dataset is no longer available for automatic download`. The canonical host `vis-www.cs.umass.edu` **fails DNS resolution from this machine even outside the sandbox**, so this is a real upstream outage, not a local network policy.

Resolution: fetched the same dataset through `sklearn.datasets.fetch_lfw_pairs(subset="10_folds", funneled=True, color=True, resize=1.0)`, which mirrors from figshare. This yields the **official 10-fold `pairs.txt`** that plan §8.2 requires, plus `lfw_funneled/` images — same data, different transport. Verified: 6,000 pairs, exactly 3,000 positive / 3,000 negative, images `(125, 94, 3)`.

Consequence for Phase 3: the LFW loader must read `data/lfw_sklearn/lfw_home/` rather than construct `torchvision.datasets.LFWPairs`. Flagging now so it is not discovered mid-Phase-3.

**2. Python 3.12, not the plan's 3.11.** This box has only system 3.12 and conda 3.13; no 3.11 is installed. `.venv` is built on 3.12.3. Nothing in the stack objected.

**3. `gdown` is an undeclared CelebA dependency.** The first CelebA attempt failed with `RuntimeError: To download files from GDrive, 'gdown' is required` — torchvision shells out to it and does not declare it. Added to `requirements.txt`. Once installed, CelebA auto-download succeeded in full (1.44 GB, no quota error), so the manual Kaggle fallback was **not** needed. It is documented in the README anyway.

**4. PKSampler drops identities with fewer than K images (default).** CelebA has 44 identities with a single image and 613 with fewer than 4; within the train split, **488 of 8,000 identities (6.1%) are unusable at K=4**, leaving 7,512.

The alternative is sampling with replacement, which manufactures a positive pair from one image repeated — distance exactly 0, zero gradient for contrastive and triplet, and a trivially-solved row for InfoNCE. That is a free, uninformative loss term that would dilute the very statistic E3 measures (fraction of active triplets). Dropping is the default; `allow_replacement=True` is available and tested. Sensitivity, for the record: K=2 costs 32 ids (0.4%), K=4 costs 488 (6.1%), K=8 costs 1,131 (14.1%).

**5. Additions beyond the brief's file list.** `tests/test_splits.py` (10 tests) and `src/data/splits.py` — the brief names only the sampler and geometry test files, but gate item 2 demands a disjointness assertion, and an identity leak is the one bug here that makes results look *better*, so it is tested rather than trusted. `pyproject.toml` was added alongside `requirements.txt` (plan §13 allows either) because `src/` must be importable from scripts, tests, and `torchrun` alike.

**6. Repo reorganized on request.** The four source documents moved to `docs/`; the poster is now `docs/poster.png`. The plan file is `docs/plan.md`, not the `siamese_infonce_implementation_plan.md` named in the protocol.

**7. GPUs 0 and 1 were freed on request.** They held a vLLM server (`google/gemma-4-31B-it`, TP=2) from the unrelated `302-ai-simulation-paper` project, idle at 0% for 2h28m. Terminated; all three GPUs now report 0 MiB used.

**No contradictions with the plan's expected outcomes** — Phase 0 has no predicted-result column to contradict.

---

## Learning checkpoint (for Ali, before approving)

1. Why must splits be identity-disjoint rather than image-disjoint — what exactly would leak into verification metrics otherwise?
2. Why does `d² = 2 − 2s` make the poster's "cosine similarity" and "Euclidean distance" boxes two views of the same thing?

---

**STOPPING HERE.** Phase 1 requires explicit approval.
