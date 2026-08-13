"""Loss tests. Phase 1 covers contrastive; Phase 2 extends to triplet and InfoNCE.

The plan §12 vectors are implemented verbatim and checked in fp32.
"""

from __future__ import annotations

import pytest
import torch

import math

from src.losses.contrastive import ContrastiveLoss, contrastive_loss_from_distances
from src.losses.infonce import InfoNCELoss
from src.losses.triplet import TripletLoss, triplet_loss_from_distances

ATOL = 1e-6
GATE_ATOL = 1e-5  # plan §12 tolerance for the InfoNCE vector


# --- plan §12 exact vectors ---------------------------------------------------


@pytest.mark.parametrize(
    ("d", "y", "expected"),
    [
        (0.5, 1, 0.25),  # positive at d=0.5 -> d^2
        (0.5, 0, 0.25),  # negative inside m=1.0 -> (1 - 0.5)^2
        (1.2, 0, 0.0),  # negative beyond the margin -> no gradient, no loss
        (0.0, 1, 0.0),  # perfectly aligned positive -> zero
    ],
)
def test_contrastive_exact_vectors_margin_1(d: float, y: int, expected: float) -> None:
    """GATE (plan §12): the four contrastive vectors at m = 1.0."""
    loss = contrastive_loss_from_distances(
        torch.tensor([d], dtype=torch.float32),
        torch.tensor([y], dtype=torch.float32),
        margin=1.0,
    )
    assert loss.item() == pytest.approx(expected, abs=ATOL)


def test_all_four_vectors_in_one_batch() -> None:
    """The same four, vectorized -- catches broadcasting bugs the scalar path hides."""
    d = torch.tensor([0.5, 0.5, 1.2, 0.0])
    y = torch.tensor([1.0, 0.0, 0.0, 1.0])
    out = contrastive_loss_from_distances(d, y, margin=1.0)
    assert torch.allclose(out, torch.tensor([0.25, 0.25, 0.0, 0.0]), atol=ATOL)


# --- shape of the loss surface ------------------------------------------------


def test_positive_branch_is_monotone_increasing_in_distance() -> None:
    """Positives are pulled with no floor: further apart is always worse."""
    d = torch.linspace(0.0, 2.0, 50)
    out = contrastive_loss_from_distances(d, torch.ones_like(d), margin=1.0)
    assert (out[1:] - out[:-1] > 0).all()


def test_negative_branch_is_zero_beyond_the_margin() -> None:
    """Past m a negative contributes exactly zero -- the point of having a margin."""
    d = torch.linspace(1.0, 2.0, 25)
    out = contrastive_loss_from_distances(d, torch.zeros_like(d), margin=1.0)
    assert torch.allclose(out, torch.zeros_like(out), atol=ATOL)


def test_negative_gradient_vanishes_beyond_the_margin() -> None:
    """Not just the value -- the gradient must vanish too, or capacity is wasted."""
    d = torch.tensor([1.5], requires_grad=True)
    contrastive_loss_from_distances(d, torch.zeros(1), margin=1.0).backward()
    assert d.grad is not None and d.grad.abs().item() == pytest.approx(0.0, abs=ATOL)

    d_inside = torch.tensor([0.5], requires_grad=True)
    contrastive_loss_from_distances(d_inside, torch.zeros(1), margin=1.0).backward()
    assert d_inside.grad is not None and d_inside.grad.item() < 0  # pushes d up


@pytest.mark.parametrize("margin", [0.25, 0.5, 1.0, 1.5])
def test_margin_sets_the_cutoff(margin: float) -> None:
    """E2 sweeps m; the cutoff must track it exactly."""
    just_inside = contrastive_loss_from_distances(
        torch.tensor([margin - 0.01]), torch.zeros(1), margin=margin
    )
    just_outside = contrastive_loss_from_distances(
        torch.tensor([margin + 0.01]), torch.zeros(1), margin=margin
    )
    assert just_inside.item() > 0
    assert just_outside.item() == pytest.approx(0.0, abs=ATOL)


# --- the module over a PK batch -----------------------------------------------


def _pk_batch(p: int = 4, k: int = 4, d: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    labels = torch.arange(p).repeat_interleave(k)
    z = torch.nn.functional.normalize(torch.randn(p * k, d), dim=-1)
    return z, labels


def test_module_returns_scalar_and_logs() -> None:
    z, labels = _pk_batch()
    loss, logs = ContrastiveLoss(margin=1.0)(z, labels)
    assert loss.ndim == 0
    assert set(logs) == {"mean_d_pos", "mean_d_neg", "frac_neg_in_margin"}
    assert 0.0 <= logs["frac_neg_in_margin"] <= 1.0


def test_module_loss_is_differentiable() -> None:
    z, labels = _pk_batch()
    z.requires_grad_(True)
    loss, _ = ContrastiveLoss(margin=1.0)(z, labels)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_identical_embeddings_per_identity_drive_positive_term_to_zero() -> None:
    """Collapse each identity to a point: mean positive distance must be 0."""
    p, k, d = 4, 4, 8
    labels = torch.arange(p).repeat_interleave(k)
    centers = torch.nn.functional.normalize(torch.randn(p, d), dim=-1)
    z = centers.repeat_interleave(k, dim=0)
    _, logs = ContrastiveLoss(margin=1.0)(z, labels)
    assert logs["mean_d_pos"] == pytest.approx(0.0, abs=1e-5)


def test_negatives_are_balanced_against_positives() -> None:
    """A P=64,K=4 batch has 384 positives and 31,872 negatives; the 83:1 imbalance
    must be corrected or the negative term dominates the gradient."""
    z, labels = _pk_batch(p=16, k=4, d=16)
    module = ContrastiveLoss(margin=1.0)
    # With balancing, mean_d_neg is an average over exactly as many negatives as
    # positives; without it the loss would barely move when positives collapse.
    loss_random, _ = module(z, labels)

    centers = torch.nn.functional.normalize(torch.randn(16, 16), dim=-1)
    z_collapsed = centers.repeat_interleave(4, dim=0)
    loss_collapsed, _ = ContrastiveLoss(margin=1.0)(z_collapsed, labels)
    assert loss_collapsed < loss_random


def test_degenerate_batch_returns_zero_with_grad() -> None:
    """A single-identity batch has no negatives; must not crash the training step."""
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1).requires_grad_(True)
    labels = torch.zeros(8, dtype=torch.long)
    loss, logs = ContrastiveLoss(margin=1.0)(z, labels)
    assert loss.item() == pytest.approx(0.0, abs=ATOL)
    loss.backward()  # must have grad_fn
    assert logs["frac_neg_in_margin"] == 0.0


@pytest.mark.parametrize("margin", [0.0, -1.0, 2.5])
def test_invalid_margin_rejected(margin: float) -> None:
    """m > 2 is unsatisfiable on the unit sphere -- catch it at construction."""
    with pytest.raises(ValueError):
        ContrastiveLoss(margin=margin)


def test_frac_neg_in_margin_is_meaningful() -> None:
    """All negatives far apart -> 0.0; all coincident -> 1.0."""
    labels = torch.tensor([0, 0, 1, 1])

    # two antipodal clusters: d_neg = 2.0, well beyond m = 1.0
    far = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    _, logs_far = ContrastiveLoss(margin=1.0)(far, labels)
    assert logs_far["frac_neg_in_margin"] == pytest.approx(0.0)

    # everything at the same point: d_neg = 0.0, all inside the margin
    near = torch.tensor([[1.0, 0.0]] * 4)
    _, logs_near = ContrastiveLoss(margin=1.0)(near, labels)
    assert logs_near["frac_neg_in_margin"] == pytest.approx(1.0)


# =============================================================================
# Triplet (plan §12 gate item 2)
# =============================================================================


@pytest.mark.parametrize(
    ("d_ap", "d_an", "expected"),
    [
        (0.3, 0.9, 0.0),  # satisfied by a wide margin -> exactly zero
        (0.8, 0.9, 0.1),  # 0.8 - 0.9 + 0.2 = 0.1
    ],
)
def test_triplet_exact_vectors_alpha_02(d_ap: float, d_an: float, expected: float) -> None:
    """GATE (plan §12): the two triplet vectors at alpha = 0.2."""
    out = triplet_loss_from_distances(
        torch.tensor([d_ap]), torch.tensor([d_an]), margin=0.2
    )
    assert out.item() == pytest.approx(expected, abs=ATOL)


def test_triplet_hinge_is_exactly_zero_once_satisfied() -> None:
    """d_an > d_ap + alpha contributes no loss AND no gradient."""
    d_ap = torch.tensor([0.3], requires_grad=True)
    d_an = torch.tensor([0.9], requires_grad=True)
    triplet_loss_from_distances(d_ap, d_an, margin=0.2).backward()
    assert d_ap.grad.abs().item() == pytest.approx(0.0, abs=ATOL)
    assert d_an.grad.abs().item() == pytest.approx(0.0, abs=ATOL)


def test_triplet_active_gradient_pulls_positive_in_and_pushes_negative_out() -> None:
    d_ap = torch.tensor([0.8], requires_grad=True)
    d_an = torch.tensor([0.9], requires_grad=True)
    triplet_loss_from_distances(d_ap, d_an, margin=0.2).backward()
    assert d_ap.grad.item() > 0   # reduce d_ap
    assert d_an.grad.item() < 0   # increase d_an


@pytest.mark.parametrize("miner", ["random", "semi-hard", "batch-hard"])
def test_triplet_module_runs_with_each_miner(miner: str) -> None:
    z, labels = _pk_batch(p=8, k=4, d=16)
    z.requires_grad_(True)
    loss, logs = TripletLoss(margin=0.2, miner=miner)(z, labels)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(logs) == {"active_fraction", "mean_d_ap", "mean_d_an", "mean_violation"}
    assert 0.0 <= logs["active_fraction"] <= 1.0
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_triplet_active_fraction_is_one_on_a_fully_violating_batch() -> None:
    """Two identities placed on top of each other: every triplet violates."""
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    _, logs = TripletLoss(margin=0.2, miner="batch-hard")(z, labels)
    assert logs["active_fraction"] == pytest.approx(1.0)


def test_triplet_active_fraction_is_zero_on_a_well_separated_batch() -> None:
    """Identities far apart, each tightly clustered: nothing left to learn."""
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loss, logs = TripletLoss(margin=0.2, miner="batch-hard")(z, labels)
    assert logs["active_fraction"] == pytest.approx(0.0)
    assert loss.item() == pytest.approx(0.0, abs=ATOL)


def test_triplet_single_identity_batch_is_safe() -> None:
    z = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1).requires_grad_(True)
    labels = torch.zeros(6, dtype=torch.long)
    loss, logs = TripletLoss()(z, labels)
    loss.backward()
    assert loss.item() == pytest.approx(0.0, abs=ATOL)
    assert logs["active_fraction"] == 0.0


def test_triplet_invalid_margin_rejected() -> None:
    with pytest.raises(ValueError, match="margin must be > 0"):
        TripletLoss(margin=0.0)


# =============================================================================
# InfoNCE (plan §12 gate item 3)
# =============================================================================


def test_infonce_exact_vector_tau_1() -> None:
    """GATE (plan §12): one anchor, s_pos = 1, one negative at s = 0, tau = 1.

        L = -log( e^1 / (e^1 + e^0) ) = -log(e / (e + 1)) = 0.31326...

    Constructed geometrically: the anchor and its positive are the same unit
    vector (s = 1); the negative is orthogonal (s = 0). Only the diagonal is
    masked, so the denominator holds exactly the positive and the one negative.
    """
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])

    loss, _ = InfoNCELoss(temperature=1.0)(z, labels)

    expected = -math.log(math.e / (math.e + 1.0))
    assert expected == pytest.approx(0.31326, abs=1e-5)

    # Anchors 0 and 1 each see (positive s=1, negative s=0) -> 0.31326 exactly.
    # Anchor 2 has no positive and is excluded from the mean.
    assert loss.item() == pytest.approx(expected, abs=GATE_ATOL)


def test_infonce_denominator_includes_the_positive() -> None:
    """The loss must be bounded below by zero and never negative.

    Excluding the positive from the denominator would let the ratio exceed 1 and
    the loss go negative -- unbounded below, and the optimizer would chase it.
    """
    for seed in range(10):
        z, labels = _pk_batch(p=6, k=4, d=16, seed=seed)
        loss, _ = InfoNCELoss(temperature=0.07)(z, labels)
        assert loss.item() >= 0.0


def test_infonce_perfect_batch_approaches_the_floor_not_zero() -> None:
    """Even a perfectly separated batch cannot reach 0 -- the positive is in
    the denominator. That floor is the point of including it."""
    p, k, d = 4, 2, 32
    labels = torch.arange(p).repeat_interleave(k)
    centers = torch.nn.functional.normalize(torch.randn(p, d), dim=-1)
    z = centers.repeat_interleave(k, dim=0)
    loss, logs = InfoNCELoss(temperature=0.07)(z, labels)
    assert loss.item() > 0.0
    assert logs["pos_rank_frac"] == pytest.approx(1.0)


def test_infonce_masks_only_the_diagonal() -> None:
    """Self-similarity (s=1) must be excluded, or it dominates the denominator."""
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    loss_with_self, _ = InfoNCELoss(temperature=1.0)(z, labels)
    # If the diagonal leaked in, the denominator would gain another e^1 term and
    # the loss would be -log(e / (2e + 1)) = 0.8697, not 0.31326.
    assert loss_with_self.item() == pytest.approx(0.31326, abs=GATE_ATOL)
    assert loss_with_self.item() != pytest.approx(0.8697, abs=1e-3)


@pytest.mark.parametrize("tau", [0.03, 0.05, 0.07, 0.1, 0.2, 0.5])
def test_infonce_finite_and_differentiable_across_the_e4_sweep(tau: float) -> None:
    z, labels = _pk_batch(p=8, k=4, d=16)
    z.requires_grad_(True)
    loss, _ = InfoNCELoss(temperature=tau)(z, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_infonce_lower_temperature_sharpens_the_gradient_on_hard_negatives() -> None:
    """Smaller tau concentrates gradient on the most similar negatives."""
    z, labels = _pk_batch(p=6, k=4, d=16)

    grads = {}
    for tau in (0.05, 0.5):
        zz = z.clone().requires_grad_(True)
        InfoNCELoss(temperature=tau)(zz, labels)[0].backward()
        grads[tau] = zz.grad.abs().max().item()

    assert grads[0.05] > grads[0.5]


def test_infonce_gradient_reaches_negatives() -> None:
    """dL_i/dz_j is nonzero for a NEGATIVE j -- the fact Phase 4 depends on.

    If negatives had no gradient, a detached cross-GPU gather would be harmless.
    It is not, and this test is why.
    """
    z = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    # Loss for anchor 0 alone; rows 4 and 5 are pure negatives for it.
    loss, _ = InfoNCELoss(temperature=0.07)(z[:2], labels[:2], z, labels, rank_offset=0)
    loss.backward()

    assert z.grad[4].abs().sum() > 0, "no gradient flowed to a negative"
    assert z.grad[5].abs().sum() > 0


def test_infonce_rank_offset_masks_the_correct_global_entry() -> None:
    """DDP correctness: with rank_offset, self-similarity is masked in the GLOBAL
    matrix. Using local indices instead leaves each anchor's own s=1 in the
    denominator -- the classic silent bug the Phase 4 gate hunts."""
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    full, _ = InfoNCELoss(temperature=0.07, seed=0)(z, labels)

    # Same computation, expressed as two "ranks" of 4 anchors each.
    per_rank = []
    for rank in range(2):
        lo, hi = rank * 4, (rank + 1) * 4
        loss, _ = InfoNCELoss(temperature=0.07, seed=0)(
            z[lo:hi], labels[lo:hi], z, labels, rank_offset=lo
        )
        per_rank.append(loss)

    assert torch.stack(per_rank).mean().item() == pytest.approx(full.item(), abs=1e-5)


def test_infonce_wrong_rank_offset_changes_the_loss() -> None:
    """Proves the previous test has teeth: offset 0 on rank 1 gives a different answer."""
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    correct, _ = InfoNCELoss(temperature=0.07, seed=0)(z[4:], labels[4:], z, labels, rank_offset=4)
    wrong, _ = InfoNCELoss(temperature=0.07, seed=0)(z[4:], labels[4:], z, labels, rank_offset=0)
    assert abs(correct.item() - wrong.item()) > 1e-4


def test_supcon_uses_every_same_identity_sample_as_positive() -> None:
    """SupCon differs from InfoNCE whenever K > 2 (more than one positive exists)."""
    z, labels = _pk_batch(p=4, k=4, d=16)
    plain, _ = InfoNCELoss(temperature=0.07, supcon=False, seed=0)(z, labels)
    supcon, _ = InfoNCELoss(temperature=0.07, supcon=True, seed=0)(z, labels)
    assert abs(plain.item() - supcon.item()) > 1e-6


def test_supcon_matches_infonce_when_k_equals_2() -> None:
    """With exactly one positive per anchor, the two objectives coincide."""
    p, d = 6, 16
    labels = torch.arange(p).repeat_interleave(2)
    z = torch.nn.functional.normalize(torch.randn(p * 2, d), dim=-1)
    plain, _ = InfoNCELoss(temperature=0.07, supcon=False, seed=0)(z, labels)
    supcon, _ = InfoNCELoss(temperature=0.07, supcon=True, seed=0)(z, labels)
    assert plain.item() == pytest.approx(supcon.item(), abs=1e-6)


def test_infonce_invalid_temperature_rejected() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        InfoNCELoss(temperature=0.0)
