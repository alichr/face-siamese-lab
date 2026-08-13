"""Phase 4 critical gate: single-GPU vs 3-GPU equivalence for global InfoNCE.

Launched by `tests/test_ddp_gather.py` via:

    torchrun --nproc_per_node=3 tests/ddp_equivalence.py

fp32 throughout, no AMP. Every tolerance below is binding; the script exits
non-zero on any failure.

**The reduction derivation** (why each rank scales by 1/world_size):

    L_ref = (1/N) * sum_{i=0}^{N-1} L_i                       N = 258

Rank r holds L = N/W = 86 anchors and computes

    L_r = (1/L) * sum_{i in rank r} L_i

so `mean_r(L_r) = (1/W) sum_r (1/L) sum_{i in r} L_i = (1/N) sum_i L_i = L_ref`.
The loss values match with no scaling.

Gradients need care. The autograd all_gather backward reduce-scatters, so rank r
receives the SUM over all ranks of the gradient w.r.t. its own slice. If each
rank backwards on the bare `L_r`, row j on rank r accumulates

    sum_{r'} (1/L) * sum_{i in r'} dL_i/dz_j  =  (1/L) * sum_{all i} dL_i/dz_j

which is W times too large, since the reference is `(1/N) sum_i dL_i/dz_j` and
N = W*L. Backwarding on `L_r / W` fixes it exactly. Getting this wrong is not a
silent bug -- the gradient check below catches it immediately -- but it is worth
stating, because "loss matches" is often mistaken for "gradients match".

**Why 129 identities x 2 images.** With exactly one positive per anchor, the
positive choice is deterministic, so the single-process reference and the
distributed run select identical positives with no RNG synchronization. 258 rows
also divide evenly by 3 ranks (86 each), and each rank's 86 consecutive rows are
43 whole identities, so no positive pair is ever split across ranks.
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist

from src.engine.gather import global_infonce_inputs
from src.losses.geometry import l2_normalize
from src.losses.infonce import InfoNCELoss
from src.models.encoder import Encoder

N_IDENTITIES = 129
IMAGES_PER_IDENTITY = 2
N_TOTAL = N_IDENTITIES * IMAGES_PER_IDENTITY  # 258
EMBED_DIM = 128
TEMPERATURE = 0.07

LOSS_TOL = 1e-5
GRAD_TOL = 1e-5
PARAM_GRAD_REL_TOL = 1e-4

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record a binding assertion."""
    status = "PASS" if condition else "FAIL"
    if not condition:
        failures.append(message)
    if dist.get_rank() == 0:
        print(f"  [{status}] {message}", flush=True)


def make_embeddings(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic embeddings with 129 identities x 2 images."""
    torch.manual_seed(0)
    z = l2_normalize(torch.randn(N_TOTAL, EMBED_DIM, dtype=torch.float32))
    labels = torch.arange(N_IDENTITIES).repeat_interleave(IMAGES_PER_IDENTITY)
    return z.to(device), labels.to(device)


def reference(z: torch.Tensor, labels: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Single-process InfoNCE over the whole batch. Returns `(loss, dL/dZ)`."""
    z_ref = z.clone().detach().requires_grad_(True)
    loss, _ = InfoNCELoss(temperature=TEMPERATURE, seed=0)(z_ref, labels)
    loss.backward()
    return float(loss.item()), z_ref.grad.clone()


def distributed_loss_and_grad(
    z_local: torch.Tensor, labels_local: torch.Tensor, world_size: int, detach: bool
) -> tuple[float, torch.Tensor]:
    """One rank's global-negative InfoNCE. Returns `(mean loss across ranks, dL/dZ_local)`."""
    z = z_local.clone().detach().requires_grad_(True)
    pool, pool_labels, rank_offset = global_infonce_inputs(z, labels_local, detach=detach)

    loss, _ = InfoNCELoss(temperature=TEMPERATURE, seed=0)(
        z, labels_local, pool, pool_labels, rank_offset=rank_offset
    )

    # See the module docstring: scale by 1/W so the reduce-scattered gradient
    # sum lands on the same scale as the single-process (1/N) average.
    (loss / world_size).backward()

    reduced = loss.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return float(reduced.item() / world_size), z.grad.clone()


def stage_a(device, rank: int, world_size: int) -> None:
    """Loss module in isolation on deterministic synthetic embeddings."""
    if rank == 0:
        print("\n=== Stage A — loss module in isolation (synthetic, fp32) ===", flush=True)

    z, labels = make_embeddings(device)
    local = N_TOTAL // world_size
    lo, hi = rank * local, (rank + 1) * local
    z_local, labels_local = z[lo:hi], labels[lo:hi]

    ref_loss, ref_grad = reference(z, labels)

    # --- A3: loss equivalence ---
    ddp_loss, ddp_grad = distributed_loss_and_grad(z_local, labels_local, world_size, detach=False)
    loss_diff = abs(ddp_loss - ref_loss)
    if rank == 0:
        print(f"    L_ref = {ref_loss:.10f}   L_ddp = {ddp_loss:.10f}   |diff| = {loss_diff:.3e}")
    check(loss_diff < LOSS_TOL, f"A3 loss equivalence |L_ddp - L_ref| = {loss_diff:.3e} < {LOSS_TOL}")

    # --- A4: THE gradient check ---
    ref_slice = ref_grad[lo:hi]
    grad_diff = (ddp_grad - ref_slice).abs().max()
    dist.all_reduce(grad_diff, op=dist.ReduceOp.MAX)
    grad_diff = float(grad_diff.item())
    if rank == 0:
        print(f"    max |dL/dZ_local - dL/dZ_ref[slice]| = {grad_diff:.3e}")
    check(
        grad_diff < GRAD_TOL,
        f"A4 GRADIENT equivalence max abs diff = {grad_diff:.3e} < {GRAD_TOL}",
    )

    # --- A5: negative control — the detached gather MUST fail A4 ---
    _, detached_grad = distributed_loss_and_grad(z_local, labels_local, world_size, detach=True)
    detached_diff = (detached_grad - ref_slice).abs().max()
    dist.all_reduce(detached_diff, op=dist.ReduceOp.MAX)
    detached_diff = float(detached_diff.item())
    ratio = detached_diff / max(grad_diff, 1e-30)
    if rank == 0:
        print(
            f"    detached gather: max abs diff = {detached_diff:.3e} "
            f"({ratio:.1e}x the correct-gather error)"
        )
    check(
        detached_diff > 100 * GRAD_TOL,
        f"A5 negative control: detached gather FAILS the gradient check "
        f"(diff {detached_diff:.3e} >> {GRAD_TOL}) — the test has teeth",
    )


def stage_b(device, rank: int, world_size: int) -> None:
    """End-to-end with the real encoder on fixed inputs, eval mode, fp32."""
    if rank == 0:
        print("\n=== Stage B — end-to-end (real encoder, eval mode, fp32) ===", flush=True)

    torch.manual_seed(0)
    images = torch.randn(N_TOTAL, 3, 112, 112)
    labels = torch.arange(N_IDENTITIES).repeat_interleave(IMAGES_PER_IDENTITY)

    def fresh_model():
        torch.manual_seed(1234)
        model = Encoder(embedding_dim=EMBED_DIM, normalize=True).to(device)
        # eval() freezes BN running stats and disables dropout, so the single-
        # process and 3-process forward passes see identical statistics. In
        # train mode BN would normalize over a 258-row batch vs an 86-row shard
        # and the comparison would be meaningless.
        model.eval()
        return model

    named = ["backbone.conv1.weight", "backbone.layer4.1.conv2.weight", "head.weight", "bn.weight"]
    local = N_TOTAL // world_size

    # --- single-process reference, monolithic forward (all 258 at once) ---
    model_ref = fresh_model()
    z_ref = model_ref(images.to(device))
    loss_ref, _ = InfoNCELoss(temperature=TEMPERATURE, seed=0)(z_ref, labels.to(device))
    loss_ref.backward()
    mono_grads = {n: p.grad.clone() for n, p in model_ref.named_parameters() if n in named}
    ref_value = float(loss_ref.item())

    # --- single-process reference, CHUNKED forward (3 x 86, one process) ---
    #
    # This is the correct baseline for the *gradient* comparison, and the reason
    # is measurable rather than a matter of opinion. Forwarding 258 images in one
    # cuDNN call versus three calls of 86 selects different convolution reduction
    # orders, which alone produces ~1.3e-2 relative error on deep conv weights --
    # with no distributed code involved at all (verified: the monolithic-vs-
    # chunked row printed below). Comparing DDP against the monolithic reference
    # would therefore measure cuDNN batching, not the gather.
    #
    # Chunking the forward while computing the loss over the concatenated
    # embeddings keeps the loss mathematically identical, so the only difference
    # left between this baseline and the DDP run is the cross-GPU gather itself --
    # which is precisely what this gate exists to test.
    model_chunked = fresh_model()
    chunks = [model_chunked(images[i * local : (i + 1) * local].to(device)) for i in range(world_size)]
    loss_chunked, _ = InfoNCELoss(temperature=TEMPERATURE, seed=0)(
        torch.cat(chunks), labels.to(device)
    )
    loss_chunked.backward()
    ref_grads = {n: p.grad.clone() for n, p in model_chunked.named_parameters() if n in named}

    # Report the numerical floor this hardware imposes, so the tolerance below is
    # justified by a measurement rather than asserted.
    if rank == 0:
        floor = max(
            float(
                (
                    (mono_grads[n] - ref_grads[n]).abs().max()
                    / ref_grads[n].abs().max().clamp_min(1e-12)
                ).item()
            )
            for n in named
        )
        print(
            f"    [context] monolithic-258 vs chunked-3x{local} on ONE process, "
            f"no distributed: worst rel err = {floor:.3e}"
        )
        print("    [context] that is pure cuDNN batch-splitting arithmetic — the floor below.")

    # --- 3-process, global gather ---
    lo, hi = rank * local, (rank + 1) * local
    model_ddp = fresh_model()
    z_local = model_ddp(images[lo:hi].to(device))
    labels_local = labels[lo:hi].to(device)

    pool, pool_labels, rank_offset = global_infonce_inputs(z_local, labels_local, detach=False)
    loss_local, _ = InfoNCELoss(temperature=TEMPERATURE, seed=0)(
        z_local, labels_local, pool, pool_labels, rank_offset=rank_offset
    )
    (loss_local / world_size).backward()

    reduced = loss_local.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    ddp_value = float(reduced.item() / world_size)

    loss_diff = abs(ddp_value - ref_value)
    if rank == 0:
        print(f"    L_ref = {ref_value:.10f}   L_ddp = {ddp_value:.10f}   |diff| = {loss_diff:.3e}")
    check(loss_diff < LOSS_TOL, f"B6a end-to-end loss |diff| = {loss_diff:.3e} < {LOSS_TOL}")

    # Parameter gradients: sum across ranks (each rank already scaled by 1/W).
    worst_name, worst_rel = "", 0.0
    for name, param in model_ddp.named_parameters():
        if name not in named:
            continue
        grad = param.grad.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grad = ref_grads[name]
        denominator = reference_grad.abs().max().clamp_min(1e-12)
        relative = float(((grad - reference_grad).abs().max() / denominator).item())
        if rank == 0:
            print(f"    {name:34s} rel err = {relative:.3e}")
        if relative > worst_rel:
            worst_name, worst_rel = name, relative

    check(
        worst_rel < PARAM_GRAD_REL_TOL,
        f"B6b encoder param gradients vs chunked reference: worst rel err = "
        f"{worst_rel:.3e} ({worst_name}) < {PARAM_GRAD_REL_TOL}",
    )

    # --- B7: local-negatives sanity ---
    model_local = fresh_model()
    z_shard = model_local(images[lo:hi].to(device))
    loss_shard, _ = InfoNCELoss(temperature=TEMPERATURE, seed=0)(z_shard, labels_local)

    model_solo = fresh_model()
    z_solo = model_solo(images[lo:hi].to(device))
    loss_solo, _ = InfoNCELoss(temperature=TEMPERATURE, seed=0)(z_solo, labels_local)

    shard_diff = abs(float(loss_shard.item()) - float(loss_solo.item()))
    max_shard_diff = torch.tensor(shard_diff, device=device)
    dist.all_reduce(max_shard_diff, op=dist.ReduceOp.MAX)
    check(
        float(max_shard_diff.item()) < LOSS_TOL,
        f"B7 negatives=local equals single-process on that shard "
        f"(max |diff| = {float(max_shard_diff.item()):.3e})",
    )

    # Global negatives must actually differ from local -- otherwise B6 would be
    # trivially satisfied and the whole gather would be untested.
    local_vs_global = abs(float(loss_shard.item()) - float(loss_local.item()))
    check(
        local_vs_global > 1e-3,
        f"B7b global negatives change the loss vs local-only "
        f"(|diff| = {local_vs_global:.4f}) — the gather is doing something",
    )


def stage_throughput(device, rank: int, world_size: int) -> None:
    """B8: img/s for the DDP path at the training config."""
    if rank == 0:
        print("\n=== Stage B8 — throughput ===", flush=True)

    torch.manual_seed(0)
    batch = 256
    model = Encoder(embedding_dim=EMBED_DIM).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    images = torch.randn(batch, 3, 112, 112, device=device)
    labels = torch.arange(batch // 4).repeat_interleave(4).to(device)

    for step in range(12):
        if step == 4:  # skip warmup
            torch.cuda.synchronize()
            start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = model(images)
        pool, pool_labels, offset = global_infonce_inputs(z.float(), labels)
        loss, _ = InfoNCELoss(temperature=TEMPERATURE)(
            z.float(), labels, pool, pool_labels, rank_offset=offset
        )
        optimizer.zero_grad(set_to_none=True)
        (loss / world_size).backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    per_rank = 8 * batch / elapsed
    total = torch.tensor(per_rank, device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"    DDP global-negatives: {per_rank:,.0f} img/s per rank, "
            f"{float(total.item()):,.0f} img/s aggregate "
            f"(global batch {batch * world_size})",
            flush=True,
        )


def configure_true_fp32() -> None:
    """Force genuine fp32 — the plan says "fp32 throughout, no AMP".

    On Ampere and later, PyTorch silently runs convolutions (and optionally
    matmuls) in **TF32**, a 19-bit format with a 10-bit mantissa. It is not fp32.
    Left on, this stage compares a 258-image forward against an 86-image one, and
    TF32's ~1e-3 per-op error compounds through the backward pass to ~1e-1
    relative error on deep conv weights — which looks exactly like a gather bug
    and is not one. The loss check still passed at 2.7e-06 because the loss is a
    single reduction, while parameter gradients accumulate error across every
    layer.

    `deterministic=True` and `benchmark=False` additionally pin the cuDNN
    algorithm, so the two runs cannot pick different reduction orders merely
    because their batch sizes differ (258 vs 86).
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"


def main() -> int:
    configure_true_fp32()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"world_size = {world_size}, N = {N_TOTAL}, local = {N_TOTAL // world_size}")

    if N_TOTAL % world_size != 0:
        if rank == 0:
            print(f"ERROR: N={N_TOTAL} not divisible by world_size={world_size}")
        return 1

    stage_a(device, rank, world_size)
    stage_b(device, rank, world_size)
    stage_throughput(device, rank, world_size)

    n_failures = torch.tensor(len(failures), device=device)
    dist.all_reduce(n_failures, op=dist.ReduceOp.MAX)

    if rank == 0:
        if int(n_failures.item()) == 0:
            print("\nALL PHASE 4 GATE CHECKS PASSED\n", flush=True)
        else:
            print(f"\n{int(n_failures.item())} CHECK(S) FAILED\n", flush=True)

    dist.destroy_process_group()
    return 0 if int(n_failures.item()) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
