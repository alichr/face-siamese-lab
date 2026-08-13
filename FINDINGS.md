# Findings — Training a Siamese Network

What this lab actually found, panel by panel, against the poster in
[`docs/poster.png`](docs/poster.png). Every verdict is backed by a figure path and a
number produced by *this* lab. Contradictions are reported, not smoothed over.

**Setup for every number below:** ResNet-18 (11.2M params, no ImageNet pretraining),
d=128, trained 30 epochs on 8,000 CelebA identities (158,931 images) with P=64 × K=4
batches. Evaluated on 1,177 **unseen** test identities: 12,000 internal pairs, plus LFW's
standard 10-fold protocol. 35 runs, `results/comparison/master_table.md`.

---

## 0 · The noise floor — read this before any other number

Six experiment arms happened to include the *same* effective configuration
(InfoNCE, τ=0.07, P=64 K=4, d=128, aug=basic). Together with the Phase 3 baseline that
gives **7 independent replicates of one config**, and therefore a measured error bar:

| metric | mean | σ | range | **2σ — the significance bar** |
|---|---|---|---|---|
| LFW accuracy | 0.9055 | 0.0026 | 0.0060 | **0.0052** |
| TAR@FAR=1e-3 | 0.4451 | 0.0300 | 0.0880 | **0.0600** |
| Recall@1 | 0.4925 | 0.0059 | 0.0163 | **0.0118** |

Two consequences that govern this whole document:

1. **Any LFW difference below ~0.005 is noise.** That erases several apparent effects.
2. **TAR@FAR=1e-3 is by far the noisiest metric here** — its 2σ is 0.06, i.e. 13% of its
   own mean. It is the most *decision-relevant* metric (§6) and the least *stable* one, so
   it needs the largest effects before it can be trusted. This is not a flaw in TAR@FAR;
   it is what happens when a statistic depends on the extreme tail of a score
   distribution estimated from 6,000 negatives.

I did not plan this replicate set — it fell out of the matrix's overlapping arms. It is
the single most useful thing in the report, because without it at least four of the
"results" below would have been over-claimed.

---

## Panel 1 · Training objective — why L2-normalize?

**Poster claims:** normalize embeddings, `ẑ = z/‖z‖₂`, for stable similarity computation.

**Verdict: UNTESTED — the experiment I built cannot answer this.**

E6 was supposed to ablate normalization. It is a **no-op**, and the numbers prove it:
`e6_norm_on` 0.9067 vs `e6_norm_off` 0.9083 — a 0.0016 gap, well inside the 0.0052 noise
floor. The cause is in the code, not the data:

```
InfoNCE(raw embeddings, scale 7.3×) = 5.0444622040
InfoNCE(L2-normalised)              = 5.0444622040
|diff|                              = 0.000e+00
```

`cosine_similarity_matrix` normalizes its own inputs, and `l2_normalize(l2_normalize(z))
== l2_normalize(z)`, so the encoder's `normalize` flag can never reach any loss. The two
E6 runs are the same experiment twice.

Testing the claim properly requires losses that consume **raw dot products** and
un-normalized Euclidean distances — a code change, not a config change. Listed in
Extensions. What the lab *can* say is narrower and still worth stating: the identity
`d² = 2 − 2s` (Panel 3) only holds on the unit sphere, so every threshold, margin and
temperature in this project is defined in terms of a normalized geometry. Detail:
[`results/phase_reports/phase_5.md`](results/phase_reports/phase_5.md) §d1.

---

## Panel 2 · How training data is formed

**Poster claims:** positive pairs = same identity under varying pose/lighting/expression;
negatives = different identities; pairs created via mining or batch sampling; balance them.

**Verdict: SUPPORTED (by construction and test, not by an experiment).**

PK sampling is what makes in-batch positives exist at all: with K=1 a batch would be P
singleton identities and **no positive pairs**. The sampler guarantees exactly P×K —
asserted over five (P,K) combinations and on CelebA-like ragged counts
(`tests/test_sampler.py`, 23 tests).

Balancing is not cosmetic. A P=64, K=4 batch contains **384 positive pairs and 31,872
negative pairs** — an 83:1 imbalance that would let the negative term dominate the
contrastive gradient outright. Negatives are subsampled to the positive count.

Two data decisions that materially shaped the results:

- **Identity-disjoint splits.** Splitting by image instead would let the encoder answer
  from memorised identity rather than transferable similarity. Asserted at the startup of
  every run, in all three directions.
- **6.1% of train identities are unusable at K=4** (488 of 8,000 have <4 images). They are
  dropped rather than sampled with replacement, because a duplicated image is a positive
  pair at distance exactly 0 — free, uninformative loss that would dilute the
  active-triplet statistic E3 exists to measure.

Figure: `notebooks/siamese_lab_tour.ipynb` §1.5 renders a live PK batch as a P×K grid.

---

## Panel 3 · Forward pass — shared weights, cosine vs Euclidean

**Poster claims:** one encoder with shared weights θ applied to both inputs; cosine
similarity s ∈ [−1,1] and Euclidean distance d ≥ 0 as two views of the comparison.

**Verdict: SUPPORTED.**

"Shared weights" is not a constraint imposed on the architecture — it *is* the
architecture. There is one `nn.Module`, called twice; encoding two inputs separately and
encoding them concatenated give identical embeddings to 1e-5
(`tests/test_encoder.py::test_two_inputs_through_one_module_equals_batched_call`).

The two boxes are provably one quantity. For normalized embeddings

$$d^2 = \|z_1-z_2\|^2 = 2 - 2s$$

verified at atol 1e-5 row-wise, on the full N×N matrix, and in fp32 as well as fp64. So
$d$ is strictly decreasing in $s$: **thresholding one is thresholding the other**, and the
poster's two boxes are notation, not two different methods.

This produced the project's **first real bug**. `sqrt(2−2s)` on the self-similarity
diagonal is `sqrt(0)`, whose derivative is infinite; the diagonal is masked out so its
upstream gradient is exactly 0, and `0 × inf = NaN` — which silently contaminated the
entire backward pass. Fixed in `geometry._safe_sqrt`; three regression tests. Detail:
[`phase_1.md`](results/phase_reports/phase_1.md) §c1.

---

## Panel 4a · Contrastive loss — effect of margin m

**Poster claims:** `L = y·d² + (1−y)·max(0, m−d)²`; y=1 pulls together, y=0 pushes apart
by at least m.

**Verdict: SUPPORTED, with the optimum outside the swept range.**

Figure: `results/comparison/v7_e2.png` · Numbers: E2, four runs

| m | LFW | alignment | uniformity |
|---|---|---|---|
| 0.25 | 0.7798 ± 0.0165 | 0.0075 | −0.0889 |
| 0.5 | 0.8033 ± 0.0189 | 0.0277 | −0.3783 |
| 1.0 | 0.8278 ± 0.0164 | 0.1018 | −1.2348 |
| **1.5** | **0.8453 ± 0.0171** | 0.2159 | −2.0025 |

The span is 0.0655 LFW — **12× the noise floor**, so the trend is unambiguous. Small m is
clearly worse, exactly as predicted.

The *mechanism* is visible in the diagnostics, and it is more interesting than the
accuracy column. At m=0.25 the model reaches alignment **0.0075** — essentially perfect
positive clustering — with uniformity **−0.089**, which is nearly the theoretical worst
(0). That is textbook **embedding collapse**: once every negative is more than 0.25 away,
the negative term switches off entirely and nothing resists everything piling into one
region. Alignment alone would score this run as near-perfect. It is the worst model in the
experiment.

Caveat: LFW rises monotonically to m=1.5 with no turnover, so the sweep does not bracket a
maximum. Since d ≤ 2 on the unit sphere, m must be ≤ 2; the useful range 1.5–2.0 went
unexplored.

---

## Panel 4b · Triplet loss — margin α and mining

**Poster claims:** `L = max(0, d_ap − d_an + α)`; use hard or semi-hard negatives.
**Plan §10 predicted:** mining matters more than α; random collapses; *semi-hard and
batch-hard keep learning*.

**Verdict: NUANCED — the headline is emphatically supported, the batch-hard half is REFUTED.**

Figure: `results/comparison/v7_e3.png` · Numbers: E3, nine runs

| α | random | semi-hard | batch-hard |
|---|---|---|---|
| 0.1 | 0.8767 ± 0.0141 | **0.8980 ± 0.0146** | 0.8102 ± 0.0141 |
| 0.2 | 0.8788 ± 0.0151 | **0.8980 ± 0.0151** | 0.7877 ± 0.0165 |
| 0.4 | 0.8757 ± 0.0121 | 0.8930 ± 0.0129 | 0.8165 ± 0.0147 |

**Mining matters more than α — confirmed, decisively.** The miner spread is **0.110** LFW;
the α spread within a fixed miner is **0.005**, at or below the noise floor. That is a
**22× ratio**, and it means α is effectively a free parameter here while the miner is the
entire decision.

**Batch-hard is the worst miner, not among the best — refuted.** It sits 6–11 points below
*random* at every α. Its uniformity gives the reason: **−0.52 to −0.85**, against −2.5 to
−3.3 for the other two. Batch-hard is partially collapsing the embedding, the same failure
mode as small-m contrastive. The likely cause is exactly the one the literature warns
about: "hardest positive, hardest negative" is precisely the selection rule that a
mislabelled image wins, and CelebA identity labels are not clean. Semi-hard exists to sit
between *useless* and *destructive*, and on this dataset that middle ground is worth 11
points of LFW.

---

## Panel 4c · InfoNCE — temperature τ and number of negatives

**Poster claims:** denominator over the batch including the positive; τ ≈ 0.07; more
negatives → stronger signal.
**Plan §10 predicted:** U-shaped accuracy, best ≈ 0.05–0.1; uniformity rises as τ falls.

**Verdict: NUANCED.**

Figure: `results/comparison/v7_e4.png` · Numbers: E4, six runs

| τ | LFW | TAR@1e-3 | alignment | uniformity |
|---|---|---|---|---|
| 0.03 | 0.9032 ± 0.0126 | 0.4050 | 0.2358 | −1.4487 |
| 0.05 | 0.9037 ± 0.0124 | 0.4702 | 0.3936 | −2.2234 |
| 0.07 | 0.9035 ± 0.0141 | 0.4542 | 0.5461 | −2.9132 |
| 0.1 | 0.9037 ± 0.0147 | 0.4028 | 0.6890 | −3.4858 |
| 0.2 | 0.8970 ± 0.0145 | 0.3232 | 0.5901 | −3.3976 |
| 0.5 | 0.8607 ± 0.0184 | 0.1225 | 0.4052 | −2.7228 |

**The plateau is flat, and I over-read it at first.** Across τ ∈ {0.03, 0.05, 0.07, 0.1},
LFW spans **0.0005** — a fifth of the noise floor. These four are *indistinguishable*. My
initial reading picked τ=0.05 as a TAR@FAR peak (0.4702 vs 0.4050 at τ=0.03), but that
0.065 gap is barely above TAR@FAR's own 2σ of 0.060. **There is no resolvable optimum
inside 0.03–0.1 at this sample size.** The honest statement is a *plateau* from 0.03 to
0.1 and a cliff beyond: τ=0.2 costs 0.009 (marginal) and τ=0.5 costs **0.045** LFW and
drops TAR@FAR to 0.1225 (both far outside noise).

**Uniformity rises monotonically as τ falls — supported, and cleanly.** −3.49 at τ=0.1 →
−1.45 at τ=0.03, with alignment moving the opposite way (0.689 → 0.236). Low τ sharpens
the softmax onto the hardest negatives, tightening positives at the cost of spread. The
mechanism is exactly what Wang & Isola predict, and it is visible even where accuracy is
flat — the diagnostics resolve structure the headline metric cannot.

The τ=0.5 failure is instructive: both alignment *and* uniformity get worse (0.405,
−2.72). A flat softmax weights every negative nearly equally, so the gradient carries
almost no information about which negatives are actually confusable.

---

## Panel 5 · The training loop

**Poster claims:** sample batch → create pairs/triplets → encode with shared network →
compute similarities → compute loss → backprop → repeat.

**Verdict: SUPPORTED — and the batch similarity matrix reproduces panel 5.4 literally.**

Figure: `results/baseline_infonce/figures/v3_similarity_heatmap.png`

Sorting a batch's rows and columns by identity makes the K×K same-identity blocks land on
the diagonal. An untrained model shows no structure; the trained model shows crisp
block-diagonal structure. The notebook (§4.3 vs §7.4) puts the two side by side.

Pipeline validity was established by E0 before any comparison was run: overfitting 10
identities reached **val AUC 0.9927** with mean s_pos 0.986 vs s_neg 0.057. That single
cheap run rules out swapped labels, a broken sampler, a sign-flipped loss and an
unattached optimizer simultaneously.

Training curves: `results/*/figures/v5_training_curves.png`.

---

## Panel 6 · Face verification — thresholding, ROC, TAR@FAR

**Poster claims:** s = 0.91 > t = 0.50 → Accept; s = 0.22 < t → Reject. Monitor ROC,
verification accuracy (e.g. TAR@FAR), Recall@K.

**Verdict: SUPPORTED.**

Figures: `results/baseline_infonce/figures/v1_similarity_histograms.png`,
`results/comparison/v2_roc_e1.png`, `results/baseline_infonce/figures/verify_same.png`

The demo, on two unseen test identities:

```
000016.jpg ↔ 118259.jpg (same person)      s = 0.8861  >  t = 0.5186  →  ACCEPT
000017.jpg ↔ 034921.jpg (different people) s = -0.0303 <  t = 0.5186  →  REJECT
```

Close to the poster's illustrative 0.91 / 0.22. The threshold is *fitted* (on the internal
test pairs), not assumed — a verification system without a separately-fitted threshold is
a similarity function, not a system.

**Why TAR@FAR rather than accuracy — demonstrated, not asserted.** Two synthetic systems
differing only in 0.5% of impostors scoring like genuine users land within 5% on accuracy
but **>10 points apart on TAR@FAR=1e-3** (`tests/test_metrics.py`). The same effect is
real in E1: the three losses span 0.072 on LFW accuracy but **0.372 on TAR@FAR=1e-3**, a
5.8× wider separation.

The V1 histogram overlap region is the ROC made concrete: it is exactly the set of pairs
no threshold can classify correctly, and its area is what AUC integrates.

**Why the LFW threshold is fitted out-of-fold** — also measured, not assumed. Fitting the
threshold on the held-out fold itself inflates accuracy relative to the honest 9-fold fit
(`tests/test_metrics.py::test_out_of_fold_threshold_is_not_optimistic`; the notebook §7.3
prints the gap for the real model).

---

## Panel 7 · When should I use InfoNCE?

**Poster claims:** works best with large mini-batches (more negatives → stronger signal);
excellent for representation learning on large identity datasets; generalizes to unseen
identities; especially useful before fine-tuning.

### 7a · Does InfoNCE beat the pairwise losses? — **SUPPORTED, emphatically**

Figures: `results/comparison/v2_roc_e1.png`, `results/comparison/v6_align_uniform.png`

| E1 run | LFW | AUC | TAR@1e-3 | Recall@1 | alignment | uniformity |
|---|---|---|---|---|---|---|
| contrastive | 0.8348 ± 0.0201 | 0.9364 | 0.0780 | 0.1100 | 0.1028 | −1.2541 |
| triplet (semi-hard) | 0.8980 ± 0.0100 | 0.9662 | 0.3125 | 0.3786 | 0.5764 | −3.2641 |
| **InfoNCE** | **0.9072 ± 0.0109** | **0.9785** | **0.4495** | **0.4884** | 0.5437 | −2.8880 |

InfoNCE ≥ triplet ≥ contrastive on every metric, exactly the predicted ordering. But the
*size* of the gap depends entirely on which question you ask:

| metric | contrastive → InfoNCE | ratio |
|---|---|---|
| LFW accuracy | 0.8348 → 0.9072 | 1.09× |
| TAR@FAR=1e-3 | 0.0780 → 0.4495 | **5.8×** |
| Recall@1 | 0.1100 → 0.4884 | **4.4×** |

**This is the poster's orange caveat, quantified.** The caveat says classical pairwise
losses stay "simpler and effective when the task is purely pairwise verification" — and
on the purely pairwise metric the gap really is modest (9% relative). The moment you ask
for a *representation* — a low false-accept operating point, or retrieval against a
gallery — the gap becomes 4–6×. Both halves of the poster's claim are correct; they are
just about different metrics, and the poster does not say so.

**Why**, from V6: contrastive sits alone at alignment 0.103 / uniformity −1.254 — tight
positives, badly-spread embedding. It is partially collapsed. Triplet and InfoNCE both
reach uniformity ≈ −2.9 to −3.3. Alignment alone would rank contrastive **best**; the pair
of diagnostics is what exposes it.

### 7b · Do more negatives help? — **NUANCED: yes for the objective, not for LFW**

Figures: `results/comparison/v7_e5.png`, `results/comparison/v6_align_uniform.png`

| negatives/anchor | LFW | TAR@1e-3 | Recall@1 | alignment | uniformity |
|---|---|---|---|---|---|
| 64 | **0.9078 ± 0.0101** | 0.3177 | 0.4615 | 0.3231 | −2.2147 |
| 128 | 0.9065 ± 0.0157 | 0.4150 | 0.4807 | 0.4308 | −2.5817 |
| 256 | 0.9035 ± 0.0169 | 0.4503 | 0.4877 | 0.5468 | −2.9035 |
| **768** (3-GPU DDP) | 0.8852 ± 0.0130 | **0.4582** | 0.4393 | 0.7347 | **−3.2202** |

Three different answers from one experiment:

- **LFW accuracy: refuted.** It *falls* 0.9078 → 0.8852. The 64-vs-768 gap (0.0226) is 4×
  the noise floor, so the decline is real.
- **TAR@FAR=1e-3: supported, monotone.** 0.3177 → 0.4150 → 0.4503 → 0.4582. The
  end-to-end gain (0.1405) is 2.3× TAR@FAR's own 2σ. More negatives genuinely buy
  low-false-accept performance.
- **Uniformity: supported, perfectly monotone.** −2.21 → −2.58 → −2.90 → −3.22, with
  alignment worsening in lockstep 0.32 → 0.73. V6 shows this as a clean path through the
  plane.

**The mechanism.** More negatives per anchor is a stronger *uniformity* pressure — that is
literally what the InfoNCE denominator does — and uniformity is bought at the cost of
alignment. Mid-threshold accuracy (LFW) rewards alignment; the low-FAR tail rewards
uniformity. So "more negatives → stronger signal" is true about the *objective* and about
the metric that cares about the tail, and false about the metric that does not.

**Honest caveat.** All rows see the same number of images per epoch, so larger batches take
proportionally **fewer optimizer steps** (2,483/epoch at batch 64 vs 206 at batch 768).
Part of the LFW decline may be under-training rather than batch size. Separating the two
needs a step-matched rerun — listed in Extensions.

**This row depends on Phase 4.** The 768-negative run used the gradient-preserving
cross-GPU gather validated in [`phase_4.md`](results/phase_reports/phase_4.md): loss
agreement 4.77e-07 and **gradient** agreement 4.38e-05, with a negative control confirming
a detached gather fails the same check by 2.05e-02 — 1.8 million times worse. Without that
gate this row would have been small-batch InfoNCE wearing a large-batch label, and the
monotone TAR@FAR and uniformity trends above would have been fiction.

### 7c · Pre-train → fine-tune — **NOT TESTED**

E9 is a stretch row in plan §10 and was not run. The poster's "especially useful before
fine-tuning" claim is unevaluated here.

---

## Panel 8 · Practical tips

| Tip | Verdict | Evidence |
|---|---|---|
| Strong augmentation | **REFUTED** at this budget | E7: none 0.9038, basic 0.9027, **strong 0.8877** |
| Balance positives and negatives | **SUPPORTED** by construction | 83:1 imbalance corrected; `tests/test_losses.py` |
| Use hard or semi-hard negatives | **NUANCED** | semi-hard yes (+0.11 LFW); batch-hard **no** (−0.09 vs random) |
| Normalize embeddings | **UNTESTED** | E6 is a no-op — Panel 1 |
| Tune margin / temperature | **NUANCED** | m matters (0.066 span); α does not (0.005); τ has a wide plateau then a cliff |
| Monitor ROC, TAR@FAR, Recall@K | **SUPPORTED — the single most useful tip** | see below |

**Augmentation (E7).** none (0.9038) and basic (0.9027) are indistinguishable — 0.0011,
one fifth of the noise floor. `strong` is genuinely worse: 0.0161 below, 3× the floor.
Two plausible reasons, and this lab cannot separate them: 30 epochs may be too short for
heavy augmentation to pay off, and CelebA is *already* aligned and tightly cropped, so
colour jitter + grayscale + blur may be destroying identity cues (skin tone, hair colour)
rather than nuisance variation. Testing this needs a longer schedule — Extensions.

**Embedding dimension (E8).** d=64 (0.9098), 128 (0.9080) and 256 (0.9112) are all within
noise of each other; d=512 (0.9002) is ~0.011 below the best, 2× the floor. The plan's
"mild effect, plateau by 128–256" is confirmed; 512 appears to over-parameterize at this
data scale. **d=64 is as good as d=128 here** — a free 2× saving in storage and match
cost that the poster does not mention.

**The most useful tip in practice was "monitor TAR@FAR and Recall@K".** Every substantive
finding in this document — the 5.8× loss gap, the E5 divergence, the E4 plateau — is
invisible or misleading in LFW accuracy alone.

---

## Scoring the plan's predictions

| # | Plan §10 predicted | Outcome | Score |
|---|---|---|---|
| E1 | InfoNCE ≥ triplet ≥ contrastive; gap shrinks for pure pairwise verification | Exactly right, both halves | ✅ **hit** |
| E2 | Small m → poor separation, histograms shift | Right; no optimum bracketed | ✅ **hit** |
| E3 | Mining > α; random collapses; semi-hard **and batch-hard** keep learning | Mining ≫ α (22×) right; batch-hard **worst of three** | ⚠️ **half miss** |
| E4 | U-shaped, best ≈ 0.05–0.1; uniformity rises as τ falls | Plateau not U (0.03–0.1 indistinguishable); uniformity exactly right | ⚠️ **partial** |
| E5 | Accuracy grows with negatives, diminishing returns | LFW **falls**; TAR@FAR and uniformity rise monotonically | ⚠️ **metric-dependent** |
| E6 | Norm off → drift, instability, worse ROC | **Untestable as built** | ❌ **void** |
| E7 | Strong augmentation wins, largest effect for InfoNCE | Strong is **worst** | ❌ **miss** |
| E8 | Mild effect, plateau by 128–256 | Confirmed; 512 slightly worse | ✅ **hit** |

**3 hits, 3 partial, 1 miss, 1 void.** The prediction that failed hardest (E7) and the one
that split by metric (E5) are the two most informative rows in the matrix.

---

## Contradictions — recorded, not tuned away (plan §14.7)

1. **Batch-hard mining is the worst miner** (E3), 6–11 LFW points below random, with
   uniformity −0.52 to −0.85 indicating partial collapse. The plan grouped it with
   semi-hard as "keeps learning".
2. **Strong augmentation hurts** (E7), 0.016 LFW below no augmentation at all.
3. **More negatives lowers LFW accuracy** while raising TAR@FAR and uniformity (E5).
   Whether "more negatives help" has no metric-free answer.
4. **E4 has no resolvable optimum** in 0.03–0.1; the four values differ by a fifth of the
   noise floor. My own first reading of this data claimed a τ=0.05 peak — that claim did
   not survive computing the error bar.
5. **E6 cannot test its own hypothesis** — an implementation flaw in this lab, not a
   finding about the science.

Bugs found and fixed along the way, all by tests rather than by inspection:
**NaN from `sqrt(0)` on the masked diagonal** ([phase_1](results/phase_reports/phase_1.md)),
**NaN from `-inf × 0` in SupCon** ([phase_2](results/phase_reports/phase_2.md)),
**TF32 silently violating "fp32 throughout"** and
**cuDNN batch-splitting masquerading as a gather bug**
([phase_4](results/phase_reports/phase_4.md)).

---

## Extensions

| Extension | What it would teach |
|---|---|
| **A real normalization ablation** | Losses on raw dot products / un-normalized distances, so Panel 1's claim is finally testable. The highest-value item here: it closes the one gap in the matrix. |
| **Step-matched E5 rerun** | Equalize optimizer *steps* rather than images, separating "batch size" from "fewer updates" in the LFW decline. |
| **Longer schedule for E7** | 90–120 epochs to test whether strong augmentation is genuinely harmful on aligned faces or merely slower to pay off. |
| **SupCon** (`--supcon`, implemented and tested, never run) | Whether promoting all K−1 same-identity samples to positives beats one — the K=4 batch already contains the extra positives for free. |
| **ArcFace head** | A margin-based *classification* objective as a fourth contender; the standard way real face systems beat metric-learning losses, and the obvious "is this whole family the right choice?" control. |
| **MoCo-style memory queue** | Decouples negative count from batch size, testing whether E5's uniformity gains persist without the alignment cost. |
| **Cross-dataset evaluation on CASIA** | Every number here is CelebA-trained; a second identity distribution would test whether the loss ordering is a property of the losses or of CelebA. |
| **Label-noise audit of batch-hard** | Inspect the pairs batch-hard selects. If they are mislabelled CelebA images, that confirms the collapse mechanism directly. |

---

## The exit exam — one paragraph

**When to pick which loss.** Start with InfoNCE unless you have a reason not to: it won
every metric here and its advantage grows with how much you need a *representation* rather
than a pair score. Use triplet with **semi-hard** mining when you need the relative-ordering
objective — never batch-hard on data with label noise. Reach for contrastive only when the
task really is a single pairwise threshold and simplicity has value; it cost 9% on LFW but
**5.8×** on TAR@FAR=1e-3, so "simpler and effective" holds only for the narrow question.
**What τ and batch size buy you:** τ anywhere in 0.03–0.1 is equivalent — it is not worth
tuning — but τ ≥ 0.2 is actively harmful. Batch size buys *uniformity* and low-false-accept
performance rather than mid-threshold accuracy, so scale it if you operate at FAR ≤ 1e-3 and
do not bother if you do not. **The tip from Panel 8 that mattered most:** monitor TAR@FAR
and Recall@K. Every real result in this project is invisible or actively misleading in
accuracy alone — including two I initially got wrong by reading the accuracy column first.

---

*35 runs · 189 tests · phase reports in [`results/phase_reports/`](results/phase_reports/)
· master table in [`results/comparison/master_table.md`](results/comparison/master_table.md)
· interactive tour in [`notebooks/siamese_lab_tour.ipynb`](notebooks/siamese_lab_tour.ipynb)*
