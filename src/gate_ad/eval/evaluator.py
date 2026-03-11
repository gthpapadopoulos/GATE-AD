"""Evaluation loop."""

from __future__ import annotations

import gc
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from gate_ad.eval.metrics import image_metrics, pixel_metrics, _pro_weights_from_gt
from gate_ad.eval.scoring import mean_topk, score_single_pass, score_test_time_masking


@dataclass
class EvalConfig:
    device: str = "cuda:0"
    topk_ratio: float = 0.05

    border_patches: int = 3



    image_score_pool: str = "topk_mean"


    test_time_masking: bool = False
    test_mask_ratio: float = 0.25
    test_full_coverage: bool = True
    test_full_coverage_cap: int = 10000
    test_masks: int = 0
    a: float = 2.0
    return_image_scores: bool = False


def _crop_border(scores_2d: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return scores_2d
    h, w = scores_2d.shape
    if 2 * border >= h or 2 * border >= w:
        return scores_2d
    return scores_2d[border : h - border, border : w - border]

def _pool_image_score(scores_2d: np.ndarray, *, method: str, topk_ratio: float) -> float:
    m = str(method or "topk_mean").strip().lower()
    if scores_2d.size == 0:
        return 0.0
    if m in ("topk", "topk_mean", "mean_topk"):
        return mean_topk(torch.from_numpy(scores_2d), float(topk_ratio))
    if m in ("max", "maximum"):
        return float(np.max(scores_2d))
    raise ValueError(f"Unknown image_score_pool={method!r} (expected topk_mean|max)")


def _resize_bilinear(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    t = torch.nn.functional.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy()


def _resize_and_smooth(arr: np.ndarray, out_hw: tuple[int, int], sigma: float = 4.0) -> np.ndarray:
    resized = _resize_bilinear(arr, out_hw)
    return gaussian_filter(resized, sigma=sigma)


def _resize_nearest(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    t = torch.nn.functional.interpolate(t, size=out_hw, mode="nearest")
    return t.squeeze(0).squeeze(0).cpu().numpy()


@torch.inference_mode()
def evaluate(
    model,
    graphs: Iterable,
    gt_pixel_maps: Iterable[np.ndarray],
    is_anomaly: Iterable[bool],
    grid_sizes: Iterable[tuple[int, int]],
    image_shapes: Iterable[tuple[int, int]],
    cfg: EvalConfig,
) -> dict:
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    image_scores = []
    image_labels = []
    image_score_rows = [] if cfg.return_image_scores else None
    pro_num_regions = 0
    eval_total_time_sec = 0.0
    eval_num_images = 0

    image_shapes_list = list(image_shapes)
    grid_sizes_list = list(grid_sizes)
    gt_pixel_maps_list = list(gt_pixel_maps)
    is_anomaly_list = list(is_anomaly)
    graphs_list = list(graphs)

    total_pixels = int(sum(h * w for h, w in image_shapes_list))
    if total_pixels <= 0:
        pixel_scores = np.zeros((0,), dtype=np.float32)
        pixel_labels = np.zeros((0,), dtype=np.int32)
        pixel_weights = np.zeros((0,), dtype=np.float32)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            scores_path = os.path.join(tmpdir, "pixel_scores.dat")
            labels_path = os.path.join(tmpdir, "pixel_labels.dat")
            weights_path = os.path.join(tmpdir, "pixel_weights.dat")
            pixel_scores = np.memmap(scores_path, dtype=np.float32, mode="w+", shape=(total_pixels,))
            pixel_labels = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(total_pixels,))
            pixel_weights = np.memmap(weights_path, dtype=np.float32, mode="w+", shape=(total_pixels,))

            offset = 0
            for g, gt_map, is_anom, grid_size, img_hw in zip(
                graphs_list, gt_pixel_maps_list, is_anomaly_list, grid_sizes_list, image_shapes_list
            ):
                t0 = time.perf_counter()
                x = g.x.to(device)
                edge_index = g.edge_index.to(device)

                if cfg.test_time_masking:
                    err = score_test_time_masking(
                        model,
                        x,
                        edge_index,
                        mask_ratio=cfg.test_mask_ratio,
                        a=cfg.a,
                        full_coverage=cfg.test_full_coverage,
                        full_coverage_cap=cfg.test_full_coverage_cap,
                        test_masks=cfg.test_masks,
                    )
                else:
                    err = score_single_pass(
                        model,
                        x,
                        edge_index,
                        cfg.a,
                    )

                err = err.detach().cpu().numpy()
                hp, wp = grid_size
                patch_grid = err.reshape(hp, wp).astype(np.float32)
                patch_grid = np.nan_to_num(patch_grid, nan=0.0, posinf=0.0, neginf=0.0)

                patch_for_image = _crop_border(patch_grid, cfg.border_patches)
                img_score = _pool_image_score(
                    patch_for_image, method=cfg.image_score_pool, topk_ratio=cfg.topk_ratio
                )
                image_scores.append(img_score)
                image_labels.append(1 if is_anom else 0)
                if image_score_rows is not None:
                    image_score_rows.append(
                        {
                            "image_path": "",
                            "is_anomaly": int(bool(is_anom)),
                            "image_score": float(img_score),
                        }
                    )


                score_map = _resize_and_smooth(patch_grid, img_hw, sigma=4.0)
                score_map = np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)
                if gt_map.shape != img_hw:
                    gt_map = _resize_nearest(gt_map.astype(np.float32), img_hw)

                gt_map = np.nan_to_num(gt_map, nan=0.0, posinf=0.0, neginf=0.0)
                gt_map = (gt_map > 0).astype(np.uint8)

                n = int(img_hw[0] * img_hw[1])
                pixel_scores[offset : offset + n] = score_map.reshape(-1)
                pixel_labels[offset : offset + n] = gt_map.reshape(-1)
                w, n_regions = _pro_weights_from_gt(gt_map)
                pro_num_regions += int(n_regions)
                pixel_weights[offset : offset + n] = w.reshape(-1)
                offset += n

                eval_total_time_sec += float(time.perf_counter() - t0)
                eval_num_images += 1


            image_scores = np.asarray(image_scores, dtype=np.float32)
            image_labels = np.asarray(image_labels, dtype=np.int32)
            metrics = {}
            metrics.update(image_metrics(image_scores, image_labels))
            metrics.update(
                pixel_metrics(pixel_scores, pixel_labels, pixel_weights, num_regions=int(pro_num_regions), fpr_limit=0.3)
            )
            metrics.update(
                {
                    "eval_num_images": int(eval_num_images),
                    "eval_total_time_sec": float(eval_total_time_sec),
                    "eval_avg_time_per_image_sec": float(eval_total_time_sec / max(1, int(eval_num_images))),
                    "eval_avg_time_per_image_ms": float(1000.0 * eval_total_time_sec / max(1, int(eval_num_images))),
                }
            )
            if image_score_rows is not None:
                metrics["__image_scores__"] = image_score_rows
            return metrics

    image_scores = np.asarray(image_scores, dtype=np.float32)
    image_labels = np.asarray(image_labels, dtype=np.int32)

    metrics = {}
    metrics.update(image_metrics(image_scores, image_labels))
    metrics.update(
        pixel_metrics(pixel_scores, pixel_labels, pixel_weights, num_regions=int(pro_num_regions), fpr_limit=0.3)
    )
    metrics.update(
        {
            "eval_num_images": int(eval_num_images),
            "eval_total_time_sec": float(eval_total_time_sec),
            "eval_avg_time_per_image_sec": float(eval_total_time_sec / max(1, int(eval_num_images))),
            "eval_avg_time_per_image_ms": float(1000.0 * eval_total_time_sec / max(1, int(eval_num_images))),
        }
    )
    if image_score_rows is not None:
        metrics["__image_scores__"] = image_score_rows
    return metrics


@torch.inference_mode()
def evaluate_streaming(
    model,
    record_iter,
    total_pixels: int,
    cfg: EvalConfig,
) -> dict:
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    image_scores = []
    image_labels = []
    image_score_rows = [] if cfg.return_image_scores else None
    pro_num_regions = 0
    eval_total_time_sec = 0.0
    eval_num_images = 0

    total_pixels = int(total_pixels)
    if total_pixels <= 0:
        pixel_scores = np.zeros((0,), dtype=np.float32)
        pixel_labels = np.zeros((0,), dtype=np.int32)
        pixel_weights = np.zeros((0,), dtype=np.float32)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            scores_path = os.path.join(tmpdir, "pixel_scores.dat")
            labels_path = os.path.join(tmpdir, "pixel_labels.dat")
            weights_path = os.path.join(tmpdir, "pixel_weights.dat")
            pixel_scores = np.memmap(scores_path, dtype=np.float32, mode="w+", shape=(total_pixels,))
            pixel_labels = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(total_pixels,))
            pixel_weights = np.memmap(weights_path, dtype=np.float32, mode="w+", shape=(total_pixels,))

            offset = 0
            gc_every = int(os.environ.get("GATEAD_GC_EVERY", "0") or 0)
            for rec in record_iter:
                if isinstance(rec, (tuple, list)) and len(rec) >= 6:
                    g, gt_map, is_anom, grid_size, img_hw, image_path = rec[:6]
                else:
                    g, gt_map, is_anom, grid_size, img_hw = rec
                    image_path = ""
                t0 = time.perf_counter()
                x = g.x.to(device)
                edge_index = g.edge_index.to(device)

                if cfg.test_time_masking:
                    err = score_test_time_masking(
                        model,
                        x,
                        edge_index,
                        mask_ratio=cfg.test_mask_ratio,
                        a=cfg.a,
                        full_coverage=cfg.test_full_coverage,
                        full_coverage_cap=cfg.test_full_coverage_cap,
                        test_masks=cfg.test_masks,
                    )
                else:
                    err = score_single_pass(
                        model,
                        x,
                        edge_index,
                        cfg.a,
                    )

                err = err.detach().cpu().numpy()
                hp, wp = grid_size
                patch_grid = err.reshape(hp, wp).astype(np.float32)
                patch_grid = np.nan_to_num(patch_grid, nan=0.0, posinf=0.0, neginf=0.0)

                patch_for_image = _crop_border(patch_grid, cfg.border_patches)
                img_score = _pool_image_score(
                    patch_for_image, method=cfg.image_score_pool, topk_ratio=cfg.topk_ratio
                )
                image_scores.append(img_score)
                image_labels.append(1 if is_anom else 0)
                if image_score_rows is not None:
                    image_score_rows.append(
                        {
                            "image_path": str(image_path),
                            "is_anomaly": int(bool(is_anom)),
                            "image_score": float(img_score),
                        }
                    )

                score_map = _resize_and_smooth(patch_grid, img_hw, sigma=4.0)
                score_map = np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)
                if gt_map.shape != img_hw:
                    gt_map = _resize_nearest(gt_map.astype(np.float32), img_hw)

                gt_map = np.nan_to_num(gt_map, nan=0.0, posinf=0.0, neginf=0.0)
                gt_map = (gt_map > 0).astype(np.uint8)

                n = int(img_hw[0] * img_hw[1])
                if offset + n > total_pixels:
                    n = max(0, total_pixels - offset)
                    score_flat = score_map.reshape(-1)[:n]
                    label_flat = gt_map.reshape(-1)[:n]
                else:
                    score_flat = score_map.reshape(-1)
                    label_flat = gt_map.reshape(-1)

                pixel_scores[offset : offset + n] = score_flat
                pixel_labels[offset : offset + n] = label_flat
                w, n_regions = _pro_weights_from_gt(gt_map)
                pro_num_regions += int(n_regions)
                pixel_weights[offset : offset + n] = w.reshape(-1)[:n]
                offset += n

                eval_total_time_sec += float(time.perf_counter() - t0)
                eval_num_images += 1
                del x, edge_index, err, patch_grid, score_map, gt_map
                if gc_every > 0 and (eval_num_images % gc_every) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            image_scores = np.asarray(image_scores, dtype=np.float32)
            image_labels = np.asarray(image_labels, dtype=np.int32)
            metrics = {}
            metrics.update(image_metrics(image_scores, image_labels))
            metrics.update(
                pixel_metrics(pixel_scores, pixel_labels, pixel_weights, num_regions=int(pro_num_regions), fpr_limit=0.3)
            )
            metrics.update(
                {
                    "eval_num_images": int(eval_num_images),
                    "eval_total_time_sec": float(eval_total_time_sec),
                    "eval_avg_time_per_image_sec": float(eval_total_time_sec / max(1, int(eval_num_images))),
                    "eval_avg_time_per_image_ms": float(1000.0 * eval_total_time_sec / max(1, int(eval_num_images))),
                }
            )
            if image_score_rows is not None:
                metrics["__image_scores__"] = image_score_rows
            return metrics

    image_scores = np.asarray(image_scores, dtype=np.float32)
    image_labels = np.asarray(image_labels, dtype=np.int32)

    metrics = {}
    metrics.update(image_metrics(image_scores, image_labels))
    metrics.update(
        pixel_metrics(pixel_scores, pixel_labels, pixel_weights, num_regions=int(pro_num_regions), fpr_limit=0.3)
    )
    metrics.update(
        {
            "eval_num_images": int(eval_num_images),
            "eval_total_time_sec": float(eval_total_time_sec),
            "eval_avg_time_per_image_sec": float(eval_total_time_sec / max(1, int(eval_num_images))),
            "eval_avg_time_per_image_ms": float(1000.0 * eval_total_time_sec / max(1, int(eval_num_images))),
        }
    )
    if image_score_rows is not None:
        metrics["__image_scores__"] = image_score_rows
    return metrics
