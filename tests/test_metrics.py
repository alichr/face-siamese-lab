"""Metric tests (plan §12, Phase 3 gate item 1).

Gate assertions: perfectly separated scores -> EER = 0, AUC = 1;
shuffled labels -> AUC ~ 0.5 +/- 0.05.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.metrics.align_uniform import align_uniform, alignment, uniformity
from src.metrics.retrieval import recall_at_k
from src.metrics.verification import (
    accuracy_at,
    best_threshold,
    equal_error_rate,
    lfw_10fold_accuracy,
    tar_at_far,
    verification_metrics,
)


def _separated(n: int = 500, margin: float = 0.5):
    """Perfectly separated scores: every positive above every negative."""
    scores = np.concatenate([np.full(n, 0.5 + margin), np.full(n, 0.5 - margin)])
    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    return scores, labels


def _overlapping(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = np.concatenate([rng.normal(0.6, 0.2, n), rng.normal(0.2, 0.2, n)])
    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    return scores, labels


# --- the gate assertions ------------------------------------------------------


def test_perfect_separation_gives_eer_0_and_auc_1() -> None:
    """GATE (plan §12)."""
    scores, labels = _separated()
    metrics = verification_metrics(scores, labels)
    assert metrics["auc"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["eer"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["best_accuracy"] == pytest.approx(1.0, abs=1e-12)


def test_shuffled_labels_give_auc_near_half() -> None:
    """GATE (plan §12): AUC ~ 0.5 +/- 0.05 when the score carries no signal."""
    scores, labels = _overlapping(n=4000)
    rng = np.random.default_rng(0)
    for _ in range(5):
        shuffled = rng.permutation(labels)
        auc = verification_metrics(scores, shuffled)["auc"]
        assert 0.45 <= auc <= 0.55, f"AUC {auc} outside 0.5 +/- 0.05"


def test_perfect_separation_gives_tar_1_at_every_far() -> None:
    scores, labels = _separated()
    for target in (1e-3, 1e-2):
        tar, _ = tar_at_far(scores, labels, target)
        assert tar == pytest.approx(1.0)


# --- TAR@FAR ------------------------------------------------------------------


def test_tar_at_far_respects_the_far_budget() -> None:
    """The returned threshold must actually hold FAR at or below the target."""
    scores, labels = _overlapping()
    for target in (1e-3, 1e-2, 1e-1):
        tar, threshold = tar_at_far(scores, labels, target)
        negatives = scores[labels == 0]
        realized_far = float((negatives > threshold).mean())
        assert realized_far <= target + 1e-9, f"FAR {realized_far} exceeds {target}"
        assert 0.0 <= tar <= 1.0


def test_tar_is_monotone_in_the_far_budget() -> None:
    """A looser FAR budget can never accept fewer genuine users."""
    scores, labels = _overlapping()
    tars = [tar_at_far(scores, labels, t)[0] for t in (1e-3, 1e-2, 1e-1)]
    assert tars[0] <= tars[1] <= tars[2]


def test_tar_at_far_distinguishes_systems_accuracy_cannot() -> None:
    """Why the plan reports TAR@FAR rather than accuracy.

    Two score distributions with near-identical best accuracy but very different
    behaviour in the low-FAR regime a real system operates at.
    """
    rng = np.random.default_rng(0)
    n = 20_000

    positives = rng.normal(0.6, 0.15, n)
    negatives = rng.normal(0.2, 0.10, n)

    # System A: negatives tightly packed -- few extreme impostors.
    a_scores = np.concatenate([positives, negatives])

    # System B: identical, except 0.5% of impostors score like genuine users.
    # That contaminated 0.5% is far above the 0.1% FAR budget, so it destroys
    # TAR@FAR=1e-3 -- while touching accuracy by only ~0.25%, since accuracy
    # counts those few pairs the same as any others.
    contaminated = negatives.copy()
    n_bad = int(0.005 * n)
    contaminated[rng.choice(n, size=n_bad, replace=False)] = rng.normal(0.75, 0.05, n_bad)
    b_scores = np.concatenate([positives, contaminated])

    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)

    a = verification_metrics(a_scores, labels)
    b = verification_metrics(b_scores, labels)

    assert abs(a["best_accuracy"] - b["best_accuracy"]) < 0.05
    assert a["tar@far=0.001"] - b["tar@far=0.001"] > 0.10


# --- EER ----------------------------------------------------------------------


def test_eer_of_random_scores_is_near_half() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 4000)
    labels = rng.integers(0, 2, 4000)
    eer, _ = equal_error_rate(scores, labels)
    assert 0.42 <= eer <= 0.58


def test_eer_is_where_far_equals_frr() -> None:
    scores, labels = _overlapping()
    eer, threshold = equal_error_rate(scores, labels)
    far = float((scores[labels == 0] > threshold).mean())
    frr = float((scores[labels == 1] <= threshold).mean())
    assert abs(far - frr) < 0.02
    assert eer == pytest.approx((far + frr) / 2, abs=0.02)


# --- thresholds ---------------------------------------------------------------


def test_best_threshold_beats_any_other_threshold() -> None:
    scores, labels = _overlapping()
    threshold, accuracy = best_threshold(scores, labels)
    for candidate in np.linspace(scores.min(), scores.max(), 200):
        assert accuracy_at(scores, labels, candidate) <= accuracy + 1e-12
    assert accuracy_at(scores, labels, threshold) == pytest.approx(accuracy, abs=1e-12)


# --- LFW 10-fold --------------------------------------------------------------


def _folded(n_per_fold: int = 300, n_folds: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores, labels, folds = [], [], []
    for fold in range(n_folds):
        scores.append(rng.normal(0.6, 0.2, n_per_fold))
        labels.append(np.ones(n_per_fold))
        scores.append(rng.normal(0.2, 0.2, n_per_fold))
        labels.append(np.zeros(n_per_fold))
        folds.append(np.full(2 * n_per_fold, fold))
    return (
        np.concatenate(scores),
        np.concatenate(labels).astype(int),
        np.concatenate(folds),
    )


def test_lfw_10fold_returns_ten_accuracies_and_a_std() -> None:
    scores, labels, folds = _folded()
    out = lfw_10fold_accuracy(scores, labels, folds)
    assert len(out["folds"]) == 10
    assert 0.0 <= out["mean"] <= 1.0 and out["std"] >= 0.0
    assert out["mean"] == pytest.approx(float(np.mean(out["folds"])), abs=1e-12)


def test_lfw_10fold_is_perfect_on_separated_scores() -> None:
    n = 300
    scores, labels, folds = [], [], []
    for fold in range(10):
        scores.append(np.full(n, 0.9))
        labels.append(np.ones(n))
        scores.append(np.full(n, 0.1))
        labels.append(np.zeros(n))
        folds.append(np.full(2 * n, fold))
    out = lfw_10fold_accuracy(
        np.concatenate(scores), np.concatenate(labels).astype(int), np.concatenate(folds)
    )
    assert out["mean"] == pytest.approx(1.0)
    assert out["std"] == pytest.approx(0.0)


def test_out_of_fold_threshold_is_not_optimistic() -> None:
    """The reason for the protocol: fitting t on the test fold inflates accuracy.

    Picking the best threshold on the very pairs being scored buys accuracy that
    does not transfer. The honest out-of-fold estimate must not exceed it.
    """
    scores, labels, folds = _folded(seed=1)
    honest = lfw_10fold_accuracy(scores, labels, folds)["mean"]

    cheating = []
    for fold in np.unique(folds):
        held = folds == fold
        _, accuracy = best_threshold(scores[held], labels[held])  # fit ON the test fold
        cheating.append(accuracy)

    assert float(np.mean(cheating)) >= honest - 1e-12


# --- retrieval ----------------------------------------------------------------


def test_recall_is_perfect_when_identities_are_separated() -> None:
    torch.manual_seed(0)
    n_ids, per_id, d = 20, 5, 32
    centers = torch.nn.functional.normalize(torch.randn(n_ids, d), dim=-1)
    embeddings = centers.repeat_interleave(per_id, dim=0)
    labels = np.repeat(np.arange(n_ids), per_id)

    out = recall_at_k(embeddings, labels)
    assert out["recall@1"] == pytest.approx(1.0)
    assert out["gallery_size"] == n_ids


def test_recall_is_near_chance_for_random_embeddings() -> None:
    torch.manual_seed(0)
    n_ids, per_id, d = 50, 4, 32
    embeddings = torch.nn.functional.normalize(torch.randn(n_ids * per_id, d), dim=-1)
    labels = np.repeat(np.arange(n_ids), per_id)

    out = recall_at_k(embeddings, labels)
    assert out["recall@1"] < 0.15  # chance is 1/50 = 0.02


def test_recall_at_k_is_monotone() -> None:
    torch.manual_seed(0)
    n_ids, per_id, d = 30, 4, 16
    centers = torch.nn.functional.normalize(torch.randn(n_ids, d), dim=-1)
    noise = 0.6 * torch.randn(n_ids * per_id, d)
    embeddings = torch.nn.functional.normalize(
        centers.repeat_interleave(per_id, dim=0) + noise, dim=-1
    )
    labels = np.repeat(np.arange(n_ids), per_id)

    out = recall_at_k(embeddings, labels, ks=(1, 5, 10))
    assert out["recall@1"] <= out["recall@5"] <= out["recall@10"]


# --- alignment / uniformity ---------------------------------------------------


def test_collapse_has_perfect_alignment_and_terrible_uniformity() -> None:
    """The failure alignment alone would call a success (plan §8.4)."""
    embeddings = torch.zeros(100, 16)
    embeddings[:, 0] = 1.0  # every point identical
    labels = np.repeat(np.arange(25), 4)

    out = align_uniform(embeddings, labels)
    assert out["alignment"] == pytest.approx(0.0, abs=1e-6)
    assert out["uniformity"] == pytest.approx(0.0, abs=1e-6)  # its maximum


def test_uniform_sphere_beats_collapse_on_uniformity() -> None:
    torch.manual_seed(0)
    spread = torch.nn.functional.normalize(torch.randn(512, 16), dim=-1)
    collapsed = torch.zeros(512, 16)
    collapsed[:, 0] = 1.0
    assert uniformity(spread) < uniformity(collapsed)


def test_tight_clusters_have_lower_alignment_than_scattered_ones() -> None:
    torch.manual_seed(0)
    n_ids, per_id, d = 25, 4, 16
    centers = torch.nn.functional.normalize(torch.randn(n_ids, d), dim=-1)
    labels = np.repeat(np.arange(n_ids), per_id)

    tight = torch.nn.functional.normalize(
        centers.repeat_interleave(per_id, dim=0) + 0.05 * torch.randn(n_ids * per_id, d), dim=-1
    )
    scattered = torch.nn.functional.normalize(torch.randn(n_ids * per_id, d), dim=-1)

    assert alignment(tight, labels) < alignment(scattered, labels)


def test_alignment_requires_positive_pairs() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(10, 8), dim=-1)
    with pytest.raises(ValueError, match="no positive pairs"):
        alignment(embeddings, np.arange(10))
