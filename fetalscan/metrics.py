"""Evaluation metrics for classification and segmentation."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(y_true, y_pred, n: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, n)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    # like sklearn: average only over classes that appear in y_true or y_pred
    present = (cm.sum(0) + cm.sum(1)) > 0
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    s = pred.sum() + target.sum()
    return 1.0 if s == 0 else float(2 * inter / (s + eps))


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    union = np.logical_or(pred, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, target).sum() / union)


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-image values."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
