# Phase 2 — All losses & miners

**Status: gate passed** (with one documented, principled exception in item 5).

## (a) What was built

| Path | What |
|---|---|
| `src/losses/triplet.py` | `L = max(0, d_ap − d_an + α)`, α=0.2 default; logs active-triplet fraction |
| `src/losses/miners.py` | `random`, `semi-hard` (with fallback), `batch-hard` |
| `src/losses/infonce.py` | full N×N cosine matrix, diagonal masked, denominator over all j≠i **including the positive**; τ=0.07 fixed; `supcon` flag; `rank_offset` for Phase 4; `similarity_heatmap_data` for V3 |
| `tests/test_miners.py` | 17 tests |
| `tests/test_losses.py` | extended to 47 tests |
| `src/engine/train.py` | `build_loss` now dispatches all three losses |

## (b) Gate evidence — every vector passes exactly (fp32)

### 1. Contrastive (m=1.0) ✓
`(d=0.5,y=1)→0.25` · `(d=0.5,y=0)→0.25` · `(d=1.2,y=0)→0` · `(d=0,y=1)→0`
All four pass at atol 1e-6, individually and vectorized in one batch.

### 2. Triplet (α=0.2) ✓
`(d_ap=0.3,d_an=0.9)→0` · `(d_ap=0.8,d_an=0.9)→0.1`, atol 1e-6. Also asserted: zero loss implies **zero gradient** on both distances, and an active triplet's gradient pulls d_ap down and pushes d_an up.

### 3. InfoNCE (τ=1) ✓
```
computed = 0.31326163   expected = 0.31326169   |diff| = 5.94e-08
```
Well inside the atol 1e-5 the plan requires. Constructed geometrically — anchor and positive are the same unit vector (s=1), negative orthogonal (s=0) — so the denominator holds exactly the positive and the one negative.

### 4. Semi-hard miner ✓
Every returned negative satisfies `d_ap < d_an < d_ap + α`, asserted on a crafted 3-identity batch and again across 20 random PK batches. The documented fallback (hardest *violating* negative when the band is empty) is tested separately, so the band assertion is never quietly satisfied by vacuous cases.

### 5. Smoke test — 20 epochs of each loss on the debug set

```
contrastive        loss 0.9390 -> 0.4477   val_auc -> 0.6953
triplet random     loss 0.1934 -> 0.1245   val_auc -> 0.7424   active 0.7734 -> 0.5312
triplet semi-hard  loss 0.1307 -> 0.1577   val_auc -> 0.7089   active 0.9922 -> 0.9688
triplet batch-hard loss 0.5197 -> 0.3828   val_auc -> 0.7546   active 1.0000 -> 1.0000
infonce            loss 5.3125 -> 4.1459   val_auc -> 0.7508   gap 0.0721 -> 0.1222
```

All five complete without error. Four of five show a decreasing loss.

⚠️ **`semi-hard` triplet loss rises (0.1307 → 0.1577) while the model clearly improves.** This is not a bug and must not be tuned away. Semi-hard *selects* triplets that violate the margin, so the loss is measured on a continuously re-hardened sample: as the model improves, the miner simply finds harder cases, holding the mined loss roughly flat. The corroborating evidence that learning is real:

- val AUC rose 0.609 → 0.709
- `mean_d_ap` fell 1.357 → 1.014 (positives genuinely pulled together)
- `active_fraction` stayed at 0.97, i.e. the miner is still finding violating triplets to work on

Mined-loss magnitude is simply not a valid progress metric for an adaptive miner. Loss-vs-epoch is only comparable *within* a fixed miner.

**Bonus: E3's predicted result is already visible.** The active-triplet fraction collapses under `random` (0.77 → 0.53) while `semi-hard` and `batch-hard` hold near 1.0. That is precisely the plan's "random: active-% collapses early (wasted triplets)" prediction, at 20 epochs on 10 identities.

## (c) Surprises & deviations

**1. Second NaN of the same family, in SupCon.** ⚠️
`log_probs * positive_mask` computes `-inf × 0 = NaN`: `logits` carries `-inf` on the masked diagonal, and multiplying by a boolean mask does not remove it. Fixed with `masked_fill(~positive_mask, 0.0)`.

Generalizing the lesson from both this and the Phase 1 `sqrt(0)` bug: **masking by multiplication is only safe when the tensor is finite everywhere.** Wherever `-inf` or a singular derivative can appear, the mask must be applied with `masked_fill` (which replaces the value) rather than a product (which propagates it). Both NaN bugs were caught by gradient/finiteness assertions, not by loss values — the forward pass looked fine in both cases.

**2. Triplet loss averages over all mined triplets, not only active ones.** Averaging over actives would rescale the gradient as the active fraction falls, masking exactly the effect E3 exists to measure.

**3. Only the diagonal is masked in InfoNCE, not all same-identity entries.** With K=4, anchor *i* has 3 same-identity images present; the 2 that are not the chosen positive are treated as negatives. That is standard InfoNCE — the objective is "identify the true match among competing candidates" — and `--supcon` is the flag that changes it. Tested: SupCon and InfoNCE agree exactly when K=2 (one positive per anchor) and diverge when K=4.

**4. `rank_offset` implemented now, ahead of Phase 4.** InfoNCE takes an optional gathered pool plus the rank's row offset, because self-masking must happen at the anchor's position in the *global* matrix. Two tests already cover it: splitting a batch into two "ranks" reproduces the single-process loss to 1e-5, and using the wrong offset changes the loss measurably. This pre-positions the critical gate but does **not** pre-empt it — no distributed code was written.

**5. Console now prints every loss diagnostic.** The previous filter matched only `mean_*`/`frac_*` keys and was silently hiding `active_fraction`.

## Learning checkpoint (for Ali)

Hand-derive 0.31326. Why does including the positive in the denominator bound the loss above zero without changing what minimizes it? Why does PK sampling with K>1 make "hardest positive" meaningful at all?

---

Proceeding to Phase 3.
