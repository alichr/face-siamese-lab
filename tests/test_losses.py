"""Loss tests. Phase 1 covers contrastive; Phase 2 extends to triplet and InfoNCE.

The plan §12 vectors are implemented verbatim and checked in fp32.
"""

from __future__ import annotations

import pytest
import torch

from src.losses.contrastive import ContrastiveLoss, contrastive_loss_from_distances

ATOL = 1e-6


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
