"""Verification metrics (plan §8.1-8.2, poster panel 6 + the Tips "monitor" line).

Panel 6 is one thresholded decision: `s > t` -> Accept. Every metric here is a
different question about that decision.

  ROC-AUC        threshold-free ranking quality. Useful, but it averages over
                 operating points a real system would never choose.
  TAR@FAR        the operating point that actually matters. Fix the false-accept
                 rate to 1e-3 -- one impostor in a thousand gets in -- and ask
                 what fraction of genuine users are accepted. Plain accuracy hides
                 this: at a 50/50 pair balance, a system with 5% false accepts and
                 one with 0.1% can post the same accuracy while being worlds apart
                 for anything security-facing.
  EER            where FAR = FRR. A single summary of the crossover point.
  accuracy@t*    the poster's framing, with t* chosen on held-out data.

**The 10-fold protocol and why the threshold is fit out-of-fold.** LFW's protocol
fits t on 9 folds and tests on the 10th, rotating. Choosing t on the test fold
itself would tune a free parameter on the data being scored -- the reported
accuracy would include the gain from picking the single luckiest threshold for
those exact pairs, and would not transfer. That optimism is small but real
(typically a few tenths of a point) and it is exactly the kind of bias that makes
published numbers irreproducible.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must match")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return s, y


def tar_at_far(scores, labels, far_target: float) -> tuple[float, float]:
    """True-accept rate at a target false-accept rate.

    Returns:
        `(tar, threshold)`. The threshold is the smallest that keeps FAR at or
        below `far_target`; TAR is the genuine-accept rate there. Returns
        `(0.0, inf)` if no threshold achieves the target.
    """
    s, y = _as_arrays(scores, labels)
    far, tar, thresholds = roc_curve(y, s)

    feasible = far <= far_target
    if not feasible.any():
        return 0.0, float("inf")

    # Among thresholds meeting the FAR budget, take the one with the highest TAR.
    index = int(np.argmax(np.where(feasible, tar, -np.inf)))
    return float(tar[index]), float(thresholds[index])


def equal_error_rate(scores, labels) -> tuple[float, float]:
    """EER: the rate where FAR equals FRR (= 1 - TAR).

    Returns:
        `(eer, threshold)`, found by linear interpolation on the ROC.
    """
    s, y = _as_arrays(scores, labels)
    far, tar, thresholds = roc_curve(y, s)
    frr = 1.0 - tar

    difference = far - frr
    crossings = np.where(np.diff(np.sign(difference)) != 0)[0]
    if len(crossings) == 0:
        index = int(np.argmin(np.abs(difference)))
        return float((far[index] + frr[index]) / 2.0), float(thresholds[index])

    i = int(crossings[0])
    # Interpolate between i and i+1 where far - frr changes sign.
    denominator = difference[i + 1] - difference[i]
    t = 0.0 if denominator == 0 else -difference[i] / denominator
    eer = float(far[i] + t * (far[i + 1] - far[i]))

    # sklearn prepends `inf` as thresholds[0] (the "reject everything" point).
    # Interpolating through it gives inf - inf = NaN, which happens exactly on
    # perfectly separated scores -- the case the gate tests. Fall back to the
    # finite endpoint there.
    lo, hi = thresholds[i], thresholds[i + 1]
    if not np.isfinite(lo) or not np.isfinite(hi):
        threshold = float(hi if np.isfinite(hi) else lo)
    else:
        threshold = float(lo + t * (hi - lo))
    return eer, threshold


def best_threshold(scores, labels) -> tuple[float, float]:
    """Threshold maximizing accuracy, and that accuracy.

    Returns:
        `(threshold, accuracy)`.
    """
    s, y = _as_arrays(scores, labels)
    order = np.argsort(s)
    s_sorted, y_sorted = s[order], y[order]

    n_positive = int(y.sum())
    n_negative = len(y) - n_positive

    # Sweep the cut: everything at or below index i is rejected.
    # correct = (negatives rejected) + (positives accepted)
    negatives_below = np.cumsum(1 - y_sorted)
    positives_below = np.cumsum(y_sorted)
    correct = negatives_below + (n_positive - positives_below)

    correct = np.concatenate([[n_negative * 0 + n_positive], correct])
    candidates = np.concatenate([[s_sorted[0] - 1e-6], s_sorted])

    index = int(np.argmax(correct))
    return float(candidates[index]), float(correct[index] / len(y))


def accuracy_at(scores, labels, threshold: float) -> float:
    """Accuracy of `s > threshold` -> accept."""
    s, y = _as_arrays(scores, labels)
    return float(((s > threshold).astype(np.int64) == y).mean())


def verification_metrics(
    scores, labels, far_targets: tuple[float, ...] = (1e-3, 1e-2)
) -> dict[str, float]:
    """The full single-set verification report (plan §8.1).

    Returns:
        `auc`, `eer`, `eer_threshold`, `best_threshold`, `best_accuracy`,
        `tar@far=<t>` and `threshold@far=<t>` for each target, plus the
        positive/negative similarity means behind the V1 histogram.
    """
    s, y = _as_arrays(scores, labels)

    eer, eer_threshold = equal_error_rate(s, y)
    threshold, accuracy = best_threshold(s, y)

    out = {
        "auc": float(roc_auc_score(y, s)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "best_threshold": threshold,
        "best_accuracy": accuracy,
        "mean_s_pos": float(s[y == 1].mean()),
        "mean_s_neg": float(s[y == 0].mean()),
    }
    out["gap"] = out["mean_s_pos"] - out["mean_s_neg"]

    for target in far_targets:
        tar, tar_threshold = tar_at_far(s, y, target)
        out[f"tar@far={target:g}"] = tar
        out[f"threshold@far={target:g}"] = tar_threshold

    return out


def lfw_10fold_accuracy(scores, labels, folds) -> dict[str, float]:
    """LFW protocol: fit the threshold on 9 folds, test on the 10th (plan §8.2).

    Args:
        scores: (N,) similarity per pair.
        labels: (N,) 1 same-person, 0 different.
        folds: (N,) fold index per pair.

    Returns:
        `mean`, `std`, `folds` (per-fold accuracies) and `mean_threshold`.
        `std` is the population std over the 10 fold accuracies, the number the
        LFW protocol reports as "±".
    """
    s, y = _as_arrays(scores, labels)
    f = np.asarray(folds).ravel()

    accuracies: list[float] = []
    thresholds: list[float] = []

    for fold in np.unique(f):
        held_out = f == fold
        # Fit on everything EXCEPT the held-out fold.
        threshold, _ = best_threshold(s[~held_out], y[~held_out])
        accuracies.append(accuracy_at(s[held_out], y[held_out], threshold))
        thresholds.append(threshold)

    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "folds": [float(a) for a in accuracies],
        "mean_threshold": float(np.mean(thresholds)),
    }
