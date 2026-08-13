# Face Siamese Lab

A learning-first reimplementation of the poster **"Training a Siamese Network — Can we use InfoNCE / contrastive information loss?"** ([`docs/poster.png`](docs/poster.png)).

Every panel of the poster becomes runnable code, an experiment, and a figure — so each claim can be *seen* holding or failing, not just read. The subject is face verification: train an encoder `f_θ` so same-identity faces land close in embedding space and different identities land far apart, then measure whether **contrastive**, **triplet**, or **InfoNCE** does it best.

This is a study of *relative* comparisons between losses and hyperparameters, **not** a SOTA attempt. Expect ≈0.85–0.94 LFW accuracy from a ResNet-18 on CelebA, not the 99%+ of ArcFace-style systems trained on MS1M.

---

## Documents

| File | What it is |
|---|---|
| [`docs/plan.md`](docs/plan.md) | **The plan** — the authoritative technical spec: data, model, exact loss formulas, defaults, experiment matrix, repo layout. |
| [`docs/PHASE_BRIEFS.md`](docs/PHASE_BRIEFS.md) | The 7 phase briefs, each with its acceptance gate and a learning checkpoint. |
| [`docs/HANDOFF_PROTOCOL.md`](docs/HANDOFF_PROTOCOL.md) | How the work is executed: one phase per agent session, hard stop at every gate. |
| [`docs/poster.png`](docs/poster.png) | The original reference poster. |

Where a brief and the plan disagree, **the plan wins** (plan §14, protocol rule 5).

---

## Setup

Requires an NVIDIA GPU and driver ≥ 580 (this box: 3× RTX PRO 6000 Blackwell, 96 GB, sm_120).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Verify the GPU stack before doing anything else:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# expect: 2.13.0+cu130 3
```

> **Python version.** The plan §2 specifies Python 3.11; this box has only 3.12 (system) and 3.13 (conda base), so the venv is built on **3.12**. Recorded as a deviation in the Phase 0 report.

> **CUDA wheels.** The `--extra-index-url` line in `requirements.txt` is load-bearing. Stock PyPI torch wheels lack `sm_120` kernels and will fail at the first matmul on these cards.

### Datasets

**CelebA** (training; 202,599 images / 10,177 identities) is fetched via `torchvision.datasets.CelebA` with `target_type="identity"`.

Its Google Drive host frequently returns a quota error. If auto-download fails, the loader **stops and prints manual instructions** rather than silently substituting other data (plan §14.6). To do it manually, download the Kaggle mirror of *CelebFaces Attributes (CelebA) Dataset* and lay it out as:

```
data/celeba/
├── img_align_celeba/          # 202,599 JPGs
├── identity_CelebA.txt
├── list_attr_celeba.txt
├── list_bbox_celeba.txt
├── list_eval_partition.txt
└── list_landmarks_align_celeba.txt
```

**LFW** (evaluation only, never training) — 6,000 pairs, standard 10-fold protocol.

> ⚠️ `torchvision.datasets.LFWPairs` **no longer auto-downloads** (upstream raises "no longer available"), and the canonical host `vis-www.cs.umass.edu` does not resolve. Fetch the identical funneled dataset plus the official `pairs.txt` from scikit-learn's figshare mirror instead:
>
> ```python
> from sklearn.datasets import fetch_lfw_pairs
> fetch_lfw_pairs(subset="10_folds", funneled=True, color=True,
>                 resize=1.0, data_home="data/lfw_sklearn")
> ```

### Splits

Identity splits are **identity-disjoint**, not image-disjoint: train 8,000 ids · val 1,000 ids · test ~1,177 ids. They are generated once, committed to `data/splits/`, and asserted pairwise-disjoint at the startup of every run (plan §14.4). The fixed seeded internal eval list (6,000 positive + 6,000 negative pairs drawn from test identities) lives there too.

---

## Layout

Per plan §13:

```
configs/          one YAML per experiment (E0…E9); no hyperparameter lives only in code
data/splits/      committed identity splits + eval pair lists
src/
  data/           celeba.py · lfw.py · folder_dataset.py · pk_sampler.py · transforms.py
  models/         encoder.py            # ResNet-18 → GAP → Linear(512→d) → BN1d → L2 norm
  losses/         contrastive.py · triplet.py · infonce.py · miners.py · geometry.py
  engine/         train.py (single + DDP) · gather.py · evaluate.py
  metrics/        verification.py · retrieval.py · align_uniform.py
  viz/            histograms.py · roc.py · heatmap.py · embedding_map.py · curves.py · sweep_plots.py
scripts/          run.sh · sweep.py · compare.py · verify_pair.py
tests/            test_geometry.py · test_losses.py · test_miners.py · test_metrics.py
                  test_sampler.py · test_ddp_gather.py
results/          <exp_name>/{config.yaml, metrics.json, curves.csv, figures/, ckpts/, report.md}
```

---

## Progress

Work proceeds one phase at a time with a hard stop at each gate. Phase reports land in `results/phase_reports/phase_N.md`.

| Phase | Title | Gate | Status |
|---|---|---|---|
| 0 | Environment & data | Tests green; identity-disjoint splits committed | ✅ |
| 1 | Minimal pipeline | E0 overfit passes (val AUC 0.9927); 12.4 s/epoch | ✅ |
| 2 | All losses & miners | Every unit-test vector passes exactly | ✅ |
| 3 | Eval + viz suites | Baseline LFW 0.9023 ± 0.0165, all of V1–V6 | ✅ |
| 4 | **DDP + global negatives** | Loss 4.8e-07, **gradients** 4.4e-05, control fails by 2.0e-02 | ✅ |
| 5 | Run the experiment matrix | 35 runs, 0 failures, 87 min | ✅ |
| 6 | Findings | [`FINDINGS.md`](FINDINGS.md) | ✅ |

**Read [`FINDINGS.md`](FINDINGS.md) first** — it answers every poster claim with a figure
and a number, including the four results that contradicted the plan's predictions.
Then [`notebooks/siamese_lab_tour.ipynb`](notebooks/siamese_lab_tour.ipynb) to play with
the code interactively.

**Phase 4 is the critical gate.** Cross-GPU negative gathering fails *silently*: with a detached all-gather, training still runs and loss still falls, but gradients from other ranks' anchors never reach the local embeddings — you get small-batch InfoNCE while believing you have large-batch InfoNCE, which would invalidate experiment E5. That gate checks **gradients**, not just loss.

---

## Running

Once Phase 1 lands, every run is YAML-driven and the resolved config is copied next to its outputs:

```bash
.venv/bin/python -m src.engine.train --config configs/e1_infonce.yaml           # single GPU
.venv/bin/torchrun --nproc_per_node=3 -m src.engine.train \
    --config configs/e5_infonce_global768.yaml                                 # DDP (E5)
.venv/bin/python scripts/sweep.py --all                                        # whole matrix, 3 GPUs
.venv/bin/python scripts/compare.py                                            # master table + V2/V6/V7
.venv/bin/python -m pytest tests/                                              # 189 tests
.venv/bin/jupyter lab notebooks/siamese_lab_tour.ipynb                         # interactive tour
```

`scripts/sweep.py` keeps all 3 GPUs busy by assigning configs round-robin; `scripts/compare.py` aggregates every `metrics.json` into the cross-experiment table and figures.
