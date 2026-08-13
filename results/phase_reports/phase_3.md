# Phase 3 — Evaluation + visualization suites

**Status: gate passed.**

## (a) What was built

| Path | What |
|---|---|
| `src/data/lfw.py` | official `pairs.txt` parser (10 folds × 300 pos + 300 neg), funneled image dataset |
| `src/metrics/verification.py` | ROC-AUC, TAR@FAR, EER, best-threshold accuracy, LFW 10-fold |
| `src/metrics/retrieval.py` | Recall@{1,5,10}, 1 gallery image per identity |
| `src/metrics/align_uniform.py` | alignment + uniformity (Wang & Isola) |
| `src/viz/style.py` | shared palette + rcParams so V1–V7 read as one system |
| `src/viz/{histograms,roc,heatmap,embedding_map,curves,sweep_plots}.py` | V1–V7, PNG **and** PDF |
| `src/engine/report.py` | full eval suite + auto `report.md` |
| `scripts/verify_pair.py` | panel-6 demo |
| `tests/test_metrics.py`, `tests/test_viz.py` | 19 + 14 tests |

## (b) Gate evidence

### 1. Metric tests pass ✓
Perfect separation → EER 0.0, AUC 1.0. Shuffled labels → AUC within 0.5 ± 0.05 across 5 shuffles. Full suite: **167 tests green**.

Beyond the required vectors, two tests encode the *reasoning* the plan asks about:
- `test_out_of_fold_threshold_is_not_optimistic` — fitting the threshold on the test fold produces an accuracy at least as high as the honest out-of-fold estimate. That difference is the bias the 10-fold protocol exists to remove.
- `test_tar_at_far_distinguishes_systems_accuracy_cannot` — two systems differing only in 0.5% of impostors scoring like genuine users land within 5% on accuracy but **>10 points apart on TAR@FAR=1e-3**. That is why the plan reports TAR@FAR.

### 2. Full baseline run complete ✓

`results/baseline_infonce/` — InfoNCE, τ=0.07, 30 epochs, plan §7 defaults.

Contains `config.yaml`, `metrics.json`, `curves.csv`, `report.md`, and all of V1–V6 as PNG + PDF.

| metric | value |
|---|---|
| **LFW accuracy (10-fold)** | **0.9023 ± 0.0165** |
| internal ROC-AUC | 0.9791 |
| internal **TAR@FAR=1e-3** | **0.4882** |
| internal **TAR@FAR=1e-2** | **0.7327** |
| internal EER | 0.0758 |
| internal best-threshold accuracy | 0.9253 |
| recall@1 / @5 / @10 | 0.4952 / 0.7322 / 0.8203 |
| alignment / uniformity | 0.5437 / −2.8960 |

Runtime: 11.7 s/epoch, 13,544 img/s; best epoch 28 of 30.

### 3. Plausibility band ✓
**LFW 0.9023 > 0.80**, and inside the plan §3 expectation of 0.85–0.94. No bug indicated.

### 4. `verify_pair.py` demo ✓

```
000016.jpg ↔ 118259.jpg (same identity)
  s = 0.8861   t = 0.5186  ->  ACCEPT (same person)

000017.jpg ↔ 034921.jpg (different identities)
  s = -0.0303  t = 0.5186  ->  REJECT (different person)
```

Close to the poster's own illustrative 0.91 / 0.22. The threshold is read from the run's `metrics.json`, not invented at call time.

## (c) Surprises & deviations

**1. Figures were inspected, not just generated.** Rendering V1 revealed two real layout defects invisible to any assertion: the subtitle overprinted the title, and the direct labels collided with the x-axis label. Fixed (labels now sit on each histogram peak, title has padding). V5's end-labels collided with panel titles; fixed with y-margins. A test can confirm a PNG is non-empty; it cannot confirm it is legible.

**2. The similarity heatmap uses a diverging colour scale, not sequential.** Cosine similarity has a meaningful zero — orthogonal embeddings — so blue→gray→red puts the sign where it can be read. A sequential ramp would render s=−0.8 and s=0.0 as two shades of light, losing the distinction between "actively opposed" and "unrelated". V3 shows clear 4×4 block-diagonal structure, recreating poster panel 5.4 from real embeddings.

**3. Palette validated rather than eyeballed.** The three loss colours pass all-pairs CVD separation (worst ΔE 9.2 deutan, 24.0 normal-vision). Colour is bound to the *entity* — `infonce` is always aqua — so dropping a series from a cross-experiment plot never repaints the survivors. Tested.

**4. Retrieval is much harder than verification, as expected.** Recall@1 is 0.495 against a 399-identity gallery while pair accuracy is 0.925. Verification asks one binary question; retrieval requires outranking 398 competing identities, and a single confusable impostor anywhere in the gallery costs the query. Worth carrying into Phase 6 — the poster's panel 7 claim about "representation learning" is better tested by Recall@K than by pair accuracy.

**5. Test identities capped at 400 for diagnostics.** Retrieval, alignment/uniformity and UMAP use the first 400 test identities (7,779 query images) rather than all 1,177, to bound per-run cost across the ~35 Phase 5 runs. Verification metrics use the full committed 12,000-pair list — unchanged. The cap is identical for every run, so cross-experiment comparisons stay valid.

**6. LFW read from the sklearn mirror**, per the Phase 0 deviation. `src/data/lfw.py` parses the official `pairs.txt` and preserves positional fold membership, which the 10-fold protocol depends on.

## Learning checkpoint (for Ali)

Why is the LFW threshold fit on 9 folds and tested on the 10th? Why TAR@FAR rather than plain accuracy? What does the V1 overlap region correspond to in ROC terms?

---

Proceeding to Phase 4 — the critical gate.
