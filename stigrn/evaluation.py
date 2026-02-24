"""
STIGRN Evaluation Module

Formal quantitative evaluation of early warning signal methods:
- AUROC / AUPRC for transition detection
- Lead time measurement
- Kendall's tau for trend detection
- Bootstrap confidence intervals
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve without sklearn dependency.

    Parameters
    ----------
    scores : np.ndarray
        Continuous prediction scores (higher = more likely transition).
    labels : np.ndarray
        Binary labels (1 = transition window, 0 = stable window).

    Returns
    -------
    float
        AUROC value in [0, 1].
    """
    # Sort by descending score
    idx = np.argsort(-scores)
    sorted_labels = labels[idx]

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined, return chance level

    # Compute via trapezoidal rule on ROC curve
    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
            fpr = fp / n_neg
            tpr = tp / n_pos
            auc += (fpr - prev_fpr) * tpr
            prev_fpr = fpr

    return auc


def compute_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve.

    Parameters
    ----------
    scores : np.ndarray
        Continuous prediction scores.
    labels : np.ndarray
        Binary labels.

    Returns
    -------
    float
        AUPRC value.
    """
    idx = np.argsort(-scores)
    sorted_labels = labels[idx]

    n_pos = np.sum(labels == 1)
    if n_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    auprc = 0.0
    prev_recall = 0.0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / n_pos

        if recall > prev_recall:
            auprc += precision * (recall - prev_recall)
            prev_recall = recall

    return auprc


def label_transition_windows(
    timepoints: np.ndarray,
    bifurcation_time: float,
    pre_window: float = 0.15,
) -> np.ndarray:
    """
    Create binary labels for evaluation: windows approaching the transition
    are labeled 1, all others 0.

    Parameters
    ----------
    timepoints : np.ndarray
        Pseudotime values for each window.
    bifurcation_time : float
        Known bifurcation timepoint.
    pre_window : float
        How far before the bifurcation to label as "approaching transition".
        Windows in [bifurcation - pre_window, bifurcation] get label 1.

    Returns
    -------
    np.ndarray
        Binary labels (0 or 1).
    """
    labels = np.zeros(len(timepoints), dtype=int)
    for i, t in enumerate(timepoints):
        if (bifurcation_time - pre_window) <= t <= bifurcation_time:
            labels[i] = 1
    return labels


def compute_lead_time(
    composite_scores: np.ndarray,
    timepoints: np.ndarray,
    bifurcation_time: float,
    threshold_sigma: float = 1.5,
    min_consecutive: int = 2,
) -> Optional[float]:
    """
    Compute how far in advance of the bifurcation the warning is triggered.

    Returns
    -------
    float or None
        Lead time (positive = warning before bifurcation). None if no warning.
    """
    mu = np.mean(composite_scores)
    sigma = np.std(composite_scores)
    if sigma < 1e-12:
        return None

    threshold = mu + threshold_sigma * sigma
    above = composite_scores > threshold

    count = 0
    for i, a in enumerate(above):
        if a:
            count += 1
            if count >= min_consecutive:
                warning_time = timepoints[i - min_consecutive + 1]
                lead = bifurcation_time - warning_time
                return lead
        else:
            count = 0

    return None


def kendall_tau_trend(values: np.ndarray) -> float:
    """
    Compute Kendall's tau to assess monotonic trend in a time series.
    Values near +1 indicate increasing trend, -1 decreasing, 0 no trend.
    """
    n = len(values)
    if n < 3:
        return 0.0

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if values[j] > values[i]:
                concordant += 1
            elif values[j] < values[i]:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.0

    return (concordant - discordant) / total


def evaluate_method(
    composite_scores: np.ndarray,
    timepoints: np.ndarray,
    bifurcation_time: float,
    method_name: str = "unknown",
    pre_window: float = 0.15,
    threshold_sigma: float = 1.5,
) -> Dict:
    """
    Complete evaluation of a single EWS method on a single dataset.

    Returns
    -------
    dict with keys:
        'method', 'auroc', 'auprc', 'lead_time', 'kendall_tau',
        'warning_triggered', 'warning_timepoint'
    """
    labels = label_transition_windows(timepoints, bifurcation_time, pre_window)

    auroc = compute_auroc(composite_scores, labels)
    auprc = compute_auprc(composite_scores, labels)
    lead_time = compute_lead_time(
        composite_scores, timepoints, bifurcation_time, threshold_sigma
    )
    tau = kendall_tau_trend(composite_scores)

    return {
        "method": method_name,
        "auroc": auroc,
        "auprc": auprc,
        "lead_time": lead_time,
        "kendall_tau": tau,
        "warning_triggered": lead_time is not None,
        "n_transition_windows": int(labels.sum()),
        "n_total_windows": len(labels),
    }


def bootstrap_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap confidence interval for AUROC.

    Returns
    -------
    dict with 'mean', 'lower', 'upper', 'std'
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    aurocs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_scores = scores[idx]
        boot_labels = labels[idx]
        if boot_labels.sum() > 0 and (boot_labels == 0).sum() > 0:
            aurocs.append(compute_auroc(boot_scores, boot_labels))

    if not aurocs:
        return {"mean": 0.5, "lower": 0.5, "upper": 0.5, "std": 0.0}

    aurocs = np.array(aurocs)
    alpha = (1 - ci) / 2

    return {
        "mean": np.mean(aurocs),
        "lower": np.percentile(aurocs, alpha * 100),
        "upper": np.percentile(aurocs, (1 - alpha) * 100),
        "std": np.std(aurocs),
    }


def compare_methods(
    results: List[Dict],
) -> str:
    """
    Generate a formatted comparison table from multiple method evaluations.

    Parameters
    ----------
    results : list of dict
        Output from evaluate_method() for each method.

    Returns
    -------
    str
        Formatted table string.
    """
    header = f"{'Method':<12} {'AUROC':>7} {'AUPRC':>7} {'Lead Time':>10} {'Tau':>6} {'Warning':>8}"
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for r in results:
        lt_str = f"{r['lead_time']:.3f}" if r['lead_time'] is not None else "   N/A"
        warning_str = "YES" if r['warning_triggered'] else " no"
        line = (f"{r['method']:<12} {r['auroc']:>7.3f} {r['auprc']:>7.3f} "
                f"{lt_str:>10} {r['kendall_tau']:>6.3f} {warning_str:>8}")
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)
