"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as cc_label
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def f1_max_from_pr(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        denom = precisions + recalls
        f1_scores = (2.0 * precisions * recalls) / np.where(denom > 0, denom, np.nan)
        finite = np.isfinite(f1_scores)
        if not finite.any():
            return float("nan"), float("nan")
        idx = int(np.nanargmax(f1_scores))
        best_f1 = float(f1_scores[idx])
        best_thr = float(thresholds[idx]) if idx < len(thresholds) else float("nan")
        return best_f1, best_thr
    except Exception:
        return float("nan"), float("nan")


def _trapz_with_xmax(x: np.ndarray, y: np.ndarray, x_max: float | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    if x_max is None:
        return float(np.trapz(y, x))

    x_max = float(x_max)
    if x_max < float(x.min()) or x_max > float(x.max()):
        return float("nan")
    y_xmax = float(np.interp(x_max, x, y))
    mask = x <= x_max
    x2 = x[mask]
    y2 = y[mask]
    if x2.size == 0:
        return float("nan")
    if float(x2[-1]) != x_max:
        x2 = np.concatenate([x2, np.asarray([x_max], dtype=np.float64)])
        y2 = np.concatenate([y2, np.asarray([y_xmax], dtype=np.float64)])
    return float(np.trapz(y2, x2))


def _pro_weights_from_gt(gt_map: np.ndarray) -> tuple[np.ndarray, int]:
    gt_bin = (gt_map > 0).astype(np.uint8)
    if gt_bin.ndim != 2:
        raise ValueError(f"Expected 2D gt_map, got shape={gt_bin.shape}")
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, n_components = cc_label(gt_bin, structure=structure)
    if n_components <= 0:
        return np.zeros_like(gt_bin, dtype=np.float32), 0

    counts = np.bincount(labeled.reshape(-1))
    weights = np.zeros_like(counts, dtype=np.float32)
    if counts.size > 1:
        weights[1:] = 1.0 / np.maximum(counts[1:], 1).astype(np.float32)
    return weights[labeled].astype(np.float32, copy=False), int(n_components)


def aupro_from_flat(
    scores: np.ndarray,
    gt_labels: np.ndarray,
    gt_weights: np.ndarray,
    num_regions: int,
    fpr_limit: float = 0.3,
) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    gt_labels = np.asarray(gt_labels, dtype=np.int32)
    gt_weights = np.asarray(gt_weights, dtype=np.float32)
    if scores.shape != gt_labels.shape or scores.shape != gt_weights.shape:
        raise ValueError("scores/gt_labels/gt_weights must have the same shape")

    valid = np.isfinite(scores)
    if not valid.any():
        return float("nan")
    scores = scores[valid]
    gt_labels = gt_labels[valid]
    gt_weights = gt_weights[valid]

    num_regions = int(num_regions)
    if num_regions <= 0:
        return float("nan")
    ok = gt_labels == 0
    num_ok = int(ok.sum())
    if num_ok <= 0:
        return float("nan")

    order = np.argsort(scores, kind="quicksort")[::-1]
    scores_sorted = scores[order]
    ok_sorted = ok[order]
    pro_sorted = gt_weights[order]

    fprs = np.cumsum(ok_sorted, dtype=np.float64) / float(num_ok)
    pros = np.cumsum(pro_sorted, dtype=np.float64) / float(num_regions)

    keep = np.concatenate([np.diff(scores_sorted) != 0, np.asarray([True])])
    fprs = fprs[keep]
    pros = pros[keep]

    fprs = np.clip(np.concatenate([np.asarray([0.0]), fprs, np.asarray([1.0])]), 0.0, 1.0)
    pros = np.clip(np.concatenate([np.asarray([0.0]), pros, np.asarray([1.0])]), 0.0, 1.0)

    area = _trapz_with_xmax(fprs, pros, x_max=float(fpr_limit))
    if not np.isfinite(area):
        return float("nan")
    return float(area / float(fpr_limit))


def image_metrics(image_scores: np.ndarray, image_labels: np.ndarray) -> dict[str, float]:
    img_valid = np.isfinite(image_scores)
    if img_valid.any() and len(np.unique(image_labels[img_valid])) > 1:
        img_auroc = _safe_roc_auc(image_labels[img_valid], image_scores[img_valid])
    else:
        img_auroc = float("nan")

    if img_valid.any():
        img_ap = _safe_ap(image_labels[img_valid], image_scores[img_valid])
        img_f1, img_f1_thr = f1_max_from_pr(image_labels[img_valid], image_scores[img_valid])
    else:
        img_ap, img_f1, img_f1_thr = float("nan"), float("nan"), float("nan")

    return {
        "image_auroc": img_auroc,
        "image_ap": img_ap,
        "image_f1_max": img_f1,
        "image_f1_max_threshold": img_f1_thr,
    }


def pixel_metrics(
    pixel_scores: np.ndarray,
    pixel_labels: np.ndarray,
    pixel_weights: np.ndarray,
    num_regions: int,
    fpr_limit: float = 0.3,
) -> dict[str, float]:
    pix_valid = np.isfinite(pixel_scores)
    if pix_valid.any() and len(np.unique(pixel_labels[pix_valid])) > 1:
        pix_auroc = _safe_roc_auc(pixel_labels[pix_valid], pixel_scores[pix_valid])
    else:
        pix_auroc = float("nan")

    if pix_valid.any():
        pix_ap = _safe_ap(pixel_labels[pix_valid], pixel_scores[pix_valid])
        pix_f1, pix_f1_thr = f1_max_from_pr(pixel_labels[pix_valid], pixel_scores[pix_valid])
    else:
        pix_ap, pix_f1, pix_f1_thr = float("nan"), float("nan"), float("nan")

    pix_aupro = aupro_from_flat(pixel_scores, pixel_labels, pixel_weights, num_regions, fpr_limit)

    return {
        "pixel_auroc": pix_auroc,
        "pixel_ap": pix_ap,
        "pixel_f1_max": pix_f1,
        "pixel_f1_max_threshold": pix_f1_thr,
        "pixel_aupro": pix_aupro,
        "pixel_aupro_fpr_limit": float(fpr_limit),
        "pixel_pro_num_regions": int(num_regions),
    }
