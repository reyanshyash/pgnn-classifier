from __future__ import annotations

import math

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    positive = logits >= 0
    result = np.empty_like(logits)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def _curves(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, ...]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probabilities, kind="mergesort")
    labels = labels[order]
    probabilities = probabilities[order]
    true_positive = np.cumsum(labels == 1)
    false_positive = np.cumsum(labels == 0)
    distinct = np.where(np.diff(probabilities))[0]
    thresholds = np.r_[distinct, labels.size - 1]
    tp = np.r_[0, true_positive[thresholds]].astype(float)
    fp = np.r_[0, false_positive[thresholds]].astype(float)
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    recall = tp / max(float((labels == 1).sum()), 1.0)
    fpr = fp / max(float((labels == 0).sum()), 1.0)
    return precision, recall, fpr


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(labels), 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if selected.any():
            ece += selected.sum() / total * abs(probabilities[selected].mean() - labels[selected].mean())
    return float(ece)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    ece_bins: int = 15,
    target_precision: float = 0.90,
) -> dict[str, float | int | list[list[int]]]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    valid = np.isin(labels, [0, 1]) & np.isfinite(probabilities)
    labels, probabilities = labels[valid], np.clip(probabilities[valid], 1e-7, 1.0 - 1e-7)
    if labels.size == 0:
        raise ValueError("No valid labeled examples were provided")
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    curve_precision, curve_recall, fpr = _curves(labels, probabilities)
    has_both_classes = len(np.unique(labels)) == 2
    roc_auc = (
        float(np.sum(np.diff(fpr) * (curve_recall[1:] + curve_recall[:-1]) * 0.5))
        if has_both_classes
        else math.nan
    )
    average_precision = (
        float(np.sum(np.diff(curve_recall) * curve_precision[1:])) if (labels == 1).any() else math.nan
    )
    eligible = curve_precision >= target_precision
    recall_at_precision = float(curve_recall[eligible].max()) if eligible.any() else 0.0
    nll = float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)))

    return {
        "samples": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "roc_auc": roc_auc,
        "pr_auc": average_precision,
        f"recall_at_precision_{target_precision:.2f}": recall_at_precision,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": nll,
        "ece": expected_calibration_error(labels, probabilities, ece_bins),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    valid = np.isin(labels, [0, 1]) & np.isfinite(probabilities)
    labels, probabilities = labels[valid], probabilities[valid]
    if labels.size == 0:
        return 0.5
    candidates = np.unique(np.clip(probabilities, 0.0, 1.0))
    if candidates.size > 1000:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 1000))
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        prediction = probabilities >= threshold
        tp = np.sum(prediction & (labels == 1))
        fp = np.sum(prediction & (labels == 0))
        fn = np.sum(~prediction & (labels == 1))
        score = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def subclass_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: tuple[str, ...] = ("BD", "EB", "FP", "NTP"),
) -> dict[str, float | int | list[list[int]] | dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    valid = (labels >= 0) & (labels < len(class_names))
    labels, predictions = labels[valid], predictions[valid]
    if labels.size == 0:
        return {"samples": 0, "accuracy": 0.0, "macro_recall": 0.0, "recall_by_class": {}}
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for truth, prediction in zip(labels, predictions, strict=True):
        if 0 <= prediction < len(class_names):
            matrix[truth, prediction] += 1
    recall_by_class: dict[str, float] = {}
    observed_recalls: list[float] = []
    for index, name in enumerate(class_names):
        denominator = int(matrix[index].sum())
        if denominator > 0:
            recall = float(matrix[index, index] / denominator)
            recall_by_class[name] = recall
            observed_recalls.append(recall)
    return {
        "samples": int(labels.size),
        "accuracy": float((labels == predictions).mean()),
        "macro_recall": float(np.mean(observed_recalls)) if observed_recalls else 0.0,
        "recall_by_class": recall_by_class,
        "confusion_matrix": matrix.tolist(),
    }
