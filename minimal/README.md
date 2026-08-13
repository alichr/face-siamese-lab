# Minimal version — the whole lab in one file

[`siamese_minimal.py`](siamese_minimal.py) is ~1,000 lines (about half of that comments,
docstrings and blank lines) that you can read top to bottom in one
sitting. Same maths and the same bug fixes as [`../src/`](../src/), with the package
structure, DDP, LFW, YAML configs and 189 tests stripped out.

Use this to **understand and experiment**. Use the full lab to **run the real experiments**.

## Start here — no data, no GPU, 2 seconds

```bash
cd minimal
../.venv/bin/python siamese_minimal.py selftest
```

Checks the exact loss vectors (contrastive `0.25`, triplet `0.1`, InfoNCE `0.31326`), the
`d² = 2 − 2s` identity, the semi-hard band property, and both NaN gotchas.

```bash
../.venv/bin/python siamese_minimal.py explain
```

Six core ideas, each with a number computed live: why normalize, why PK sampling, why the
positive stays in the InfoNCE denominator, what temperature actually does, why the miner
beats the margin, and why alignment alone can be fooled.

## Then train something — ~1 minute

```bash
../.venv/bin/python siamese_minimal.py demo
```

Trains InfoNCE on 1,500 identities for 10 epochs, evaluates on **unseen** identities, and
writes `runs/infonce/summary.png` with four panels: score histograms + threshold, ROC on a
log-FAR axis, the identity-ordered similarity matrix (poster panel 5.4), and training curves.

Reaches **val AUC ≈ 0.915** in about 50 seconds on one GPU.

```bash
../.venv/bin/python siamese_minimal.py compare        # all three losses, ~3 min
../.venv/bin/python siamese_minimal.py eval  --ckpt runs/infonce/model.pt
../.venv/bin/python siamese_minimal.py verify A.jpg B.jpg --ckpt runs/infonce/model.pt
```

### What `compare` should print

```
loss                AUC      acc   TAR@1e-3      EER    align     unif
contrastive      0.8101   0.7390     0.0120   0.2675    0.194   -0.736
triplet          0.8876   0.8230     0.0610   0.1770    0.621   -2.472
infonce          0.9174   0.8345     0.2140   0.1695    0.613   -2.346
```

Two things to notice, both of which reproduce the full lab's headline result:

1. **The metric changes the story.** InfoNCE beats contrastive by **1.13×** on accuracy but
   **17.8×** on TAR@FAR=1e-3. This is the poster's orange caveat — pairwise losses really
   are "effective for purely pairwise verification" — and it is only visible if you look
   past accuracy. (The full 30-epoch lab saw 1.09× and 5.8×.)
2. **Contrastive is partially collapsed.** Its uniformity is **−0.736** against −2.4 for the
   other two, while its alignment (0.194) is the *best* of the three. Alignment alone would
   rank it first. It is last on every accuracy metric. That is exactly why both diagnostics
   get reported.

## Reading order

The file is ordered by dependency, which is also the order the ideas build:

| § | What | The idea |
|---|---|---|
| 1 | geometry | `d² = 2 − 2s` — cosine and Euclidean are one quantity |
| 2 | data | identity splits; PK batches are what make positives *exist* |
| 3 | model | "shared weights" = one module called twice |
| 4 | losses | contrastive, triplet + 3 miners, InfoNCE |
| 5 | metrics | AUC / TAR@FAR / EER, and why the choice changes the story |
| 6 | train | the loop |
| 7 | plots | what to look at |
| 8 | cli | the commands |

## Things to try

```bash
# Contrastive with too small a margin -> embedding collapse.
# Watch alignment go to ~0 (looks perfect!) while uniformity goes to ~0 (worst possible).
python siamese_minimal.py train --loss contrastive --margin 0.25
python siamese_minimal.py eval --ckpt runs/contrastive/model.pt

# The miner matters far more than the margin. Compare active_fraction.
python siamese_minimal.py train --loss triplet --miner random
python siamese_minimal.py train --loss triplet --miner semi-hard
python siamese_minimal.py train --loss triplet --miner batch-hard   # expect the WORST

# Temperature: 0.03-0.1 are equivalent; 0.5 is actively harmful.
python siamese_minimal.py train --loss infonce --tau 0.5

# Break it on purpose: K=1 means no positives exist. Read the assertion.
python siamese_minimal.py train --k 1
```

## GOTCHAs — two real NaN bugs, kept in

Both cost real debugging time in the full lab, both are marked in the source, and both are
covered by `selftest`.

**1. `sqrt(0)` on the masked diagonal.** `d = sqrt(2−2s)` and the self-similarity diagonal
is exactly 0, where `sqrt`'s derivative is infinite. Every loss masks that diagonal, so its
upstream gradient is exactly 0 — and `0 × inf = NaN`, which poisons the whole backward
pass while every forward value looks fine. Fix needs **both** a clamp before the sqrt and a
`masked_fill` after.

**2. `-inf × 0` in SupCon.** Masking by multiplication (`log_probs * mask`) is only safe
when the tensor is finite everywhere. `logits` carries `-inf` on the masked diagonal.
Use `masked_fill`.

The general lesson: both bugs were caught by **gradient finiteness assertions**, never by
looking at loss values. The forward pass was correct in both cases.

## What was cut, and where to find it

| Cut | Where it lives | Why it matters |
|---|---|---|
| DDP + cross-GPU negatives | `../src/engine/gather.py`, `../tests/ddp_equivalence.py` | The project's most important idea: a detached all-gather silently discards negatives' gradients — loss still falls, curves look healthy, and you get small-batch InfoNCE while believing otherwise. See [`../results/phase_reports/phase_4.md`](../results/phase_reports/phase_4.md). |
| LFW 10-fold protocol | `../src/metrics/verification.py` | Out-of-fold threshold fitting; the minimal version fits the threshold on the same pairs it scores, which is slightly optimistic. |
| Retrieval (Recall@K) | `../src/metrics/retrieval.py` | Much harder than verification — 0.495 vs 0.925 in the full lab. |
| UMAP, PDF export, per-run `report.md` | `../src/viz/`, `../src/engine/report.py` | |
| 189 tests | `../tests/` | |
| The 35-run matrix | `../FINDINGS.md` | The actual results, with a measured noise floor. |

**Numbers here are not comparable to the full lab's** — this trains on 1,500 identities for
10 epochs, against 8,000 for 30. Expect AUC ≈ 0.91 here versus 0.979 there. The *orderings*
between losses should still reproduce; the absolute values will not.

## After this

- [`../FINDINGS.md`](../FINDINGS.md) — every poster claim with a verdict, a figure and a
  number, including four results that contradicted the plan.
- [`../notebooks/siamese_lab_tour.ipynb`](../notebooks/siamese_lab_tour.ipynb) — the same
  ground, cell by cell, with knobs.
- [`../docs/plan.md`](../docs/plan.md) — the full spec.
