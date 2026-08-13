# Phase 4 — DDP + global negatives ⚠️ CRITICAL GATE

**Status: ALL GATE CHECKS PASSED.**

## (a) What was built

| Path | What |
|---|---|
| `src/engine/gather.py` | gradient-preserving all-gather (`torch.distributed.nn.functional.all_gather`), plus a deliberately-detached variant used only as the negative control |
| `src/engine/train.py` | DDP path: `torchrun`, SyncBN, `negatives: local \| global`, rank-0-only eval/checkpoint/logging |
| `src/data/pk_sampler.py` | rank-aware identity partitioning (`rank`, `world_size`) |
| `tests/ddp_equivalence.py` | the two-stage equivalence test, run under `torchrun --nproc_per_node=3` |
| `tests/test_ddp_gather.py` | pytest wrapper + single-process invariants (5 tests) |
| `configs/e5_global_ddp.yaml` | the one DDP job in the matrix |

## (b) Gate evidence — raw output

```
world_size = 3, N = 258, local = 86

=== Stage A — loss module in isolation (synthetic, fp32) ===
    L_ref = 6.2868895531   L_ddp = 6.2868887583   |diff| = 7.947e-07
  [PASS] A3 loss equivalence |L_ddp - L_ref| = 7.947e-07 < 1e-05
    max |dL/dZ_local - dL/dZ_ref[slice]| = 1.118e-08
  [PASS] A4 GRADIENT equivalence max abs diff = 1.118e-08 < 1e-05
    detached gather: max abs diff = 2.047e-02 (1.8e+06x the correct-gather error)
  [PASS] A5 negative control: detached gather FAILS the gradient check
         (diff 2.047e-02 >> 1e-05) — the test has teeth

=== Stage B — end-to-end (real encoder, eval mode, fp32) ===
    [context] monolithic-258 vs chunked-3x86 on ONE process, no distributed:
              worst rel err = 1.265e-02
    [context] that is pure cuDNN batch-splitting arithmetic — the floor below.
    L_ref = 5.5494923592   L_ddp = 5.5494918823   |diff| = 4.768e-07
  [PASS] B6a end-to-end loss |diff| = 4.768e-07 < 1e-05
    backbone.conv1.weight              rel err = 6.422e-06
    backbone.layer4.1.conv2.weight     rel err = 1.422e-05
    head.weight                        rel err = 4.375e-05
    bn.weight                          rel err = 3.872e-05
  [PASS] B6b encoder param gradients vs chunked reference:
         worst rel err = 4.375e-05 (head.weight) < 0.0001
  [PASS] B7 negatives=local equals single-process on that shard (max |diff| = 0.000e+00)
  [PASS] B7b global negatives change the loss vs local-only (|diff| = 1.1062)

=== Stage B8 — throughput ===
    DDP global-negatives: 15,875 img/s per rank, 47,641 img/s aggregate
    (global batch 768)

ALL PHASE 4 GATE CHECKS PASSED
```

**Throughput (gate item 8):** single-GPU 13,544 img/s (Phase 3 baseline) vs 3-GPU DDP 47,641 img/s aggregate — 3.5× on 3 GPUs. The super-linear figure is because the DDP measurement uses synthetic in-memory tensors with no dataloader, so it isolates compute; the real single-GPU number includes JPEG decoding.

## (c) The two things that went wrong, and why they were not the gather

Both were found by the gate, and neither was fixed by relaxing a tolerance.

### 1. TF32 is not fp32

Stage B initially failed with 10% relative error on `backbone.layer4.1.conv2.weight`, while the *loss* matched to 2.7e-06. That pattern — loss right, gradients wrong — is exactly what a broken gather looks like, so it deserved a real diagnosis.

Cause: on Ampere and later, PyTorch runs convolutions in **TF32** by default (19 bits, 10-bit mantissa). The plan specifies "fp32 throughout, no AMP", and TF32 silently violates it. The loss check survived because a loss is a single reduction, while parameter gradients accumulate error through every layer. Disabling TF32 (`configure_true_fp32`) cut the error ~8×, to 1.26e-2.

### 2. cuDNN batch-splitting arithmetic — the confound

1.26e-2 remained. Rather than adjust the tolerance, I ran a control with **no distributed code at all**: one GPU, one process, forwarding 258 images as a single batch versus three chunks of 86, with the loss computed over the concatenated embeddings so it is mathematically identical.

```
--- TF32=False (single GPU, no distributed, identical loss) ---
  backbone.conv1.weight            max_rel = 1.388e-03
  backbone.layer4.1.conv2.weight   max_rel = 1.265e-02
  head.weight                      max_rel = 1.463e-04
  bn.weight                        max_rel = 1.176e-04
```

Those match the DDP numbers (1.389e-03, 1.264e-02) to three significant figures. **The entire residual was cuDNN selecting different convolution reduction orders for different batch sizes. The gather contributed nothing measurable.**

The fix was to remove the confound, not to widen the gate: Stage B6b now compares DDP against a single-process reference that forwards in the *same* 86-image chunks. Loss stays mathematically identical, batch geometry is matched, and the only remaining difference is the cross-GPU gather — which is what the gate is for. Worst error dropped to **4.375e-05**, comfortably inside the plan's 1e-4. The monolithic-vs-chunked floor is printed in the gate output every run, so the tolerance is justified by a live measurement rather than asserted.

## (d) The classic bugs, checked explicitly

| Bug | Status |
|---|---|
| detached `all_gather` killing gradients | **A4 catches it**; A5 proves so — the detached variant is off by 2.047e-02, 1.8M× the correct-gather error |
| diagonal masked with *local* indices after gathering | `rank_offset = rank × local_size` passed into `InfoNCELoss`; two Phase 2 tests already cover it, including one showing a wrong offset changes the loss |
| positive index not offset into the global matrix | same mechanism — the positive mask is built from *global* labels and self-masked at `rank_offset` |
| mean-of-means with unequal anchor counts | every rank holds exactly `N/W = 86`; the sampler enforces P per rank, and the script refuses to run if `N % world_size != 0` |
| embeddings normalized *after* gathering | the encoder normalizes before returning; `global_infonce_inputs` gathers already-normalized rows and documents why |
| BN statistics differing between the two forwards | Stage B runs in **eval mode** (frozen running stats); **SyncBN** is used for real DDP training, and `train.py` prints which is active |

## (e) Surprises & deviations

**1. The `1/world_size` backward scaling is required and is not obvious.** The autograd all_gather backward reduce-scatters, so each rank receives the *sum* over ranks of the gradient w.r.t. its slice. Backwarding on the bare local loss gives gradients exactly `world_size` times too large — while the **loss value still matches perfectly**. This is a second way to get "loss right, gradients wrong", and only the gradient check finds it. Derived in the `ddp_equivalence.py` docstring and applied in the training loop.

**2. PK sampling had to become rank-aware.** With independent per-rank shuffles the same identity can land on two ranks, so some of the "extra negatives" gained by gathering would actually be positives — quietly weakening the very effect E5 measures. Every rank now draws from one shared seeded permutation and takes a disjoint slice. Tested (`test_pk_sampler_ranks_get_disjoint_identities`).

**3. DDP epoch length had to be divided by world_size.** Each rank runs `batches_per_epoch` steps, so a 3-rank epoch initially consumed 3× the images of a single-GPU epoch. E5's batch-size comparison would then also have been a training-budget comparison. Now 206 batches × 256 × 3 = 158,208 images/epoch, matching single-GPU's 158,720.

**4. The gate is now part of `pytest tests/`**, not a command to remember, and the wrapper asserts the negative control actually ran — otherwise A4 could pass vacuously.

## Learning checkpoint (for Ali)

Write out ∂Lᵢ/∂z_j for a *negative* j and observe it is nonzero:

    ∂Lᵢ/∂z_j = (1/τ)·p_ij·z_i,   p_ij = softmax_j(s_ij/τ)

With a detached gather that gradient is discarded. What objective are you then actually optimizing, and why does the loss still decrease?

---

Proceeding to Phase 5 — the experiment matrix.
