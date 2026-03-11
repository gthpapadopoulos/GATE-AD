"""Run a single object experiment (graph model only)."""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import numpy as np
import torch

from gate_ad.backbones import BackboneConfig, get_backbone
from gate_ad.data.mvtec import get_test_records as mvtec_test_records, get_train_normals as mvtec_train_normals
from gate_ad.data.visa import get_train_normals as visa_train_normals, load_visa_split
from gate_ad.data.transforms import (
    color_jitter_aug,
    hflip_image,
    random_rotate_aug,
    rotate_aug,
    vflip_image,
)
from gate_ad.eval.evaluator import EvalConfig, evaluate_streaming
from gate_ad.graph.builder import build_grid_graph
from gate_ad.models.autoencoder import PatchGraphAutoencoder
from gate_ad.training.trainer import TrainConfig, train_graphs
from gate_ad.utils.io import read_image_rgb, read_mask_gray
from gate_ad.utils.seed import set_all_seeds



FIXED_BORDER_PATCHES = 3
FIXED_GRID_K = 8
FIXED_GAT_HEADS = 1


def _default_resolution_for_model(model_name: str) -> int:
    mn = str(model_name).lower()
    if mn.startswith("dinov2"):
        return 488

    return 512


def _default_topk_ratio_for_dataset(dataset: str) -> float:
    ds = str(dataset).lower()
    if ds == "mvtec":
        return 0.025
    if ds == "visa":
        return 0.01
    return 0.05


def _default_image_score_pool_for_dataset(dataset: str) -> str:
    return "max" if str(dataset).lower() == "mpdd" else "topk_mean"


def default_rotate_aug_mode(dataset: str, object_name: str) -> str:
    ds = str(dataset).strip().lower()
    obj = str(object_name).strip().lower()
    if ds == "visa" and obj == "macaroni2":
        return "all"
    if ds == "mvtec" and obj == "screw":
        return "all"
    if ds == "mpdd" and obj in {"bracket_brown", "bracket_white", "tubes"}:
        return "all"
    return "none"


def parse_args():
    p = argparse.ArgumentParser(description="Run a single object experiment.")
    p.add_argument("--dataset", required=True, choices=["mvtec", "visa", "mpdd"])
    p.add_argument("--object_name", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--visa_split_csv", default="")

    p.add_argument("--model_name", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="If omitted: 512 for DINOv3, 488 for DINOv2.",
    )
    p.add_argument("--backbone_last_n_layers", type=int, default=8)
    p.add_argument("--backbone_layer_agg", type=str, default="avg")
    p.add_argument(
        "--backbone_layer_ids",
        type=str,
        default="",
        help=(
            "Comma-separated explicit transformer block indices (e.g., 6,7,8). "
            "If set, these are used instead of --backbone_last_n_layers."
        ),
    )
    p.add_argument(
        "--backbone_ckpt",
        type=str,
        default="",
        help="Optional path to checkpoint file for the selected backbone.",
    )

    p.add_argument("--shots", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mask_ratio", type=float, default=0.2)
    p.add_argument("--a", type=float, default=2.0, help="SCE exponent.")

    p.add_argument("--gnn_layers", type=int, default=3)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no_mlp", action="store_true", default=False)
    p.add_argument("--gnn_hidden_dims", type=str, default="", help="Comma-separated GNN hidden dims.")
    p.add_argument(
        "--gat_self_loops",
        action="store_true",
        default=False,
        help="Enable self-loops inside GATConv so each node can attend to itself.",
    )
    p.add_argument("--no_pred_head", action="store_true", default=False)
    p.add_argument("--no_target_proj", action="store_true", default=False)
    p.add_argument(
        "--gnn_residual",
        action="store_true",
        default=False,
        help="Enable long skip from first to last GNN layer output.",
    )

    p.add_argument(
        "--image_score_pool",
        type=str,
        default="auto",
        choices=["auto", "topk_mean", "max"],
        help="How to pool patch scores into an image score.",
    )

    p.add_argument("--rotate_aug_mode", type=str, default="auto", choices=["auto", "none", "all"])
    p.add_argument("--augment_hflip", action="store_true", default=False)
    p.add_argument("--augment_vflip", action="store_true", default=False)
    p.add_argument("--augment_random_rotation", action="store_true", default=False)
    p.add_argument("--augment_random_rotation_n", type=int, default=2)
    p.add_argument("--augment_color_jitter", action="store_true", default=False)
    p.add_argument("--augment_color_jitter_n", type=int, default=1)

    p.add_argument("--test_time_masking", action="store_true", default=False)
    p.add_argument("--no_test_time_masking", action="store_true", default=False)
    p.add_argument("--test_mask_ratio", type=float, default=0.2)
    p.add_argument("--test_full_coverage_cap", type=int, default=10000)
    p.add_argument(
        "--topk_ratio",
        type=float,
        default=None,
        help="If omitted: mvtec=0.025, visa=0.01, mpdd=0.05.",
    )

    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--save_image_scores_csv", action="store_true", default=False)
    p.add_argument("--image_scores_csv_name", type=str, default="image_anomaly_scores.csv")
    return p.parse_args()


def _set_backbone_env(last_n: int, agg: str, layer_ids: str = ""):
    os.environ["DINO_LAST_N_LAYERS"] = str(int(last_n))
    os.environ["DINO_LAYER_AGG"] = str(agg)
    layer_ids = str(layer_ids).strip()
    if layer_ids:
        os.environ["DINO_LAYER_IDS"] = layer_ids
    else:
        os.environ.pop("DINO_LAYER_IDS", None)


def _make_graphs_from_images(backbone, images: List[np.ndarray]):
    graphs = []
    grid_sizes = []
    for img in images:
        img_tensor, grid_size = backbone.prepare_image(img)
        feats = backbone.extract_features(img_tensor)
        feats = feats.clone().detach()
        g, grid_size = build_grid_graph(feats, grid_size, border_patches=0, grid_k=FIXED_GRID_K)
        graphs.append(g)
        grid_sizes.append(grid_size)
    return graphs, grid_sizes


def _total_pixels_from_records(test_records) -> int:
    total = 0
    for rec in test_records:
        img = read_image_rgb(rec.image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        total += int(h * w)
    return int(total)


def _dataset_for_loader(dataset: str) -> str:

    if str(dataset).lower() == "mpdd":
        return "mvtec"
    return str(dataset).lower()


def _load_train_paths(args: argparse.Namespace) -> list[str]:
    ds = _dataset_for_loader(args.dataset)
    if ds == "visa":
        return visa_train_normals(
            args.visa_split_csv,
            args.data_root,
            args.object_name,
            args.shots,
            args.seed,
            selection="first",
        )
    return mvtec_train_normals(
        args.data_root,
        args.object_name,
        args.shots,
        args.seed,
        selection="first",
    )


def _load_test_records(args: argparse.Namespace):
    ds = _dataset_for_loader(args.dataset)
    if ds == "visa":
        _train, test_records = load_visa_split(args.visa_split_csv, args.data_root, args.object_name)
        return test_records
    return mvtec_test_records(args.data_root, args.object_name)


def run_single(args: argparse.Namespace) -> int:
    if getattr(args, "resolution", None) is None:
        args.resolution = _default_resolution_for_model(args.model_name)
    if getattr(args, "topk_ratio", None) is None:
        args.topk_ratio = _default_topk_ratio_for_dataset(args.dataset)
    pool = str(getattr(args, "image_score_pool", "auto")).strip().lower()
    if pool in {"", "auto"}:
        args.image_score_pool = _default_image_score_pool_for_dataset(args.dataset)
    else:
        args.image_score_pool = pool
    rotate_mode = str(getattr(args, "rotate_aug_mode", "auto")).strip().lower()
    if rotate_mode in {"", "auto"}:
        args.rotate_aug_mode = default_rotate_aug_mode(args.dataset, args.object_name)
    else:
        args.rotate_aug_mode = rotate_mode

    set_all_seeds(args.seed, args.device)
    _set_backbone_env(
        args.backbone_last_n_layers,
        args.backbone_layer_agg,
        getattr(args, "backbone_layer_ids", ""),
    )
    backbone_ckpt = str(getattr(args, "backbone_ckpt", "")).strip()
    if backbone_ckpt:
        backbone_ckpt = os.path.abspath(backbone_ckpt)
        if not os.path.isfile(backbone_ckpt):
            raise FileNotFoundError(f"backbone_ckpt not found: {backbone_ckpt}")
        args.backbone_ckpt = backbone_ckpt
        mn = str(args.model_name).lower()
        if mn.startswith("dinov2"):
            os.environ["GATEAD_DINOV2_CKPT"] = backbone_ckpt
        elif mn.startswith("dinov3"):
            os.environ["GATEAD_DINOV3_CKPT"] = backbone_ckpt

    backbone = get_backbone(
        BackboneConfig(
            model_name=args.model_name,
            device=args.device,
            smaller_edge_size=args.resolution,
            half_precision=False,
        )
    )


    train_paths = _load_train_paths(args)

    train_images = []
    aug_rng = np.random.RandomState(int(args.seed))
    for p in train_paths:
        img = read_image_rgb(p)
        if img is None:
            continue
        base_images = rotate_aug(img) if args.rotate_aug_mode == "all" else [img]
        augmented = []
        for bimg in base_images:
            augmented.append(bimg)
            if args.augment_hflip:
                augmented.append(hflip_image(bimg))
            if args.augment_vflip:
                augmented.append(vflip_image(bimg))
            if args.augment_hflip and args.augment_vflip:
                augmented.append(vflip_image(hflip_image(bimg)))
            if args.augment_random_rotation:
                augmented.extend(
                    random_rotate_aug(
                        bimg,
                        aug_rng,
                        n=max(0, int(args.augment_random_rotation_n)),
                    )
                )
            if args.augment_color_jitter:
                augmented.extend(
                    color_jitter_aug(
                        bimg,
                        aug_rng,
                        n=max(0, int(args.augment_color_jitter_n)),
                    )
                )
        train_images.extend(augmented)
    train_num_images = int(len(train_images))

    t_train_feat0 = time.perf_counter()
    train_graphs_list, _ = _make_graphs_from_images(backbone, train_images)
    train_feature_time_sec = float(time.perf_counter() - t_train_feat0)

    in_dim = int(train_graphs_list[0].x.shape[1])
    gnn_hidden_dims = None
    if args.gnn_hidden_dims:
        gnn_hidden_dims = [int(x) for x in args.gnn_hidden_dims.split(",") if x.strip() != ""]
        if len(gnn_hidden_dims) != args.gnn_layers:
            raise ValueError("gnn_hidden_dims length must match gnn_layers")

    model = PatchGraphAutoencoder(
        in_dim=in_dim,
        gnn_layers=args.gnn_layers,
        gnn_hidden_dims=gnn_hidden_dims,
        gat_heads=FIXED_GAT_HEADS,
        gat_self_loops=bool(args.gat_self_loops),
        latent_dim=args.latent_dim,
        use_pred_head=not args.no_pred_head,
        use_target_proj=not args.no_target_proj,
        use_mlp=not args.no_mlp,
        use_residual=args.gnn_residual,
        dropout=args.dropout,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        a=args.a,
        device=args.device,
    )
    t_train0 = time.perf_counter()
    model = train_graphs(model, train_graphs_list, train_cfg)
    train_total_time_sec = float(time.perf_counter() - t_train0)


    test_records = _load_test_records(args)
    test_feature_time_sec = [0.0]
    total_pixels = _total_pixels_from_records(test_records)

    def _iter_records():
        for rec in test_records:
            img = read_image_rgb(rec.image_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            t0 = time.perf_counter()
            img_tensor, grid_size = backbone.prepare_image(img)
            feats = backbone.extract_features(img_tensor)
            test_feature_time_sec[0] += float(time.perf_counter() - t0)
            g, grid_size = build_grid_graph(feats, grid_size, border_patches=0, grid_k=FIXED_GRID_K)

            if rec.is_anomaly and rec.mask_path:
                gt = read_mask_gray(rec.mask_path)
                if gt is None:
                    print(f"WARNING: Missing/invalid GT mask: {rec.mask_path}")
                    gt = np.zeros((img_h, img_w), dtype=np.uint8)
            else:
                if rec.is_anomaly and not rec.mask_path:
                    print(f"WARNING: Anomalous sample without mask: {rec.image_path}")
                gt = np.zeros((img_h, img_w), dtype=np.uint8)

            yield g, gt, bool(rec.is_anomaly), grid_size, (img_h, img_w), rec.image_path

    eval_cfg = EvalConfig(
        device=args.device,
        topk_ratio=args.topk_ratio,
        border_patches=FIXED_BORDER_PATCHES,
        image_score_pool=args.image_score_pool,
        test_time_masking=(not args.no_test_time_masking) and bool(args.test_time_masking),
        test_mask_ratio=args.test_mask_ratio,
        test_full_coverage_cap=args.test_full_coverage_cap,
        a=args.a,
        return_image_scores=bool(args.save_image_scores_csv),
    )
    metrics = evaluate_streaming(model, _iter_records(), total_pixels, eval_cfg)
    image_score_rows = metrics.pop("__image_scores__", None)

    metrics["train_num_images"] = int(train_num_images)
    metrics["train_feature_extract_time_sec"] = float(train_feature_time_sec)
    metrics["train_total_time_sec"] = float(train_total_time_sec)
    metrics["train_wall_time_sec"] = float(train_feature_time_sec + train_total_time_sec)
    if train_num_images > 0:
        metrics["train_feature_extract_time_per_image_sec"] = float(train_feature_time_sec / train_num_images)
        metrics["train_total_time_per_image_sec"] = float(train_total_time_sec / train_num_images)
        metrics["train_wall_time_per_image_sec"] = float(
            (train_feature_time_sec + train_total_time_sec) / train_num_images
        )
    metrics["feature_extract_time_sec"] = float(test_feature_time_sec[0])
    eval_num = int(metrics.get("eval_num_images", 0))
    if eval_num > 0:
        metrics["feature_extract_time_per_image_sec"] = float(test_feature_time_sec[0] / eval_num)
    if "eval_total_time_sec" in metrics:
        metrics["inference_total_time_sec"] = float(test_feature_time_sec[0] + float(metrics["eval_total_time_sec"]))
        if eval_num > 0:
            metrics["inference_time_per_image_sec"] = float(
                metrics["inference_total_time_sec"] / eval_num
            )


    metrics["dataset"] = args.dataset
    metrics["object_name"] = args.object_name
    metrics["shots"] = int(args.shots)
    metrics["seed"] = int(args.seed)
    metrics["shot_selection"] = "first"
    metrics["model"] = args.model_name
    metrics["loss_type"] = "sce"
    metrics["a"] = float(args.a)
    metrics["topk_ratio"] = float(args.topk_ratio)
    metrics["border_patches"] = int(FIXED_BORDER_PATCHES)
    metrics["image_score_pool"] = str(args.image_score_pool)
    metrics["gnn_layers"] = int(args.gnn_layers)
    metrics["gnn_conv_type"] = "gat"
    metrics["gat_heads"] = int(FIXED_GAT_HEADS)
    metrics["gat_self_loops"] = int(bool(args.gat_self_loops))
    metrics["mask_ratio"] = float(args.mask_ratio)
    metrics["dropout"] = float(args.dropout)
    metrics["grid_k"] = int(FIXED_GRID_K)
    metrics["backbone_last_n_layers"] = int(args.backbone_last_n_layers)
    metrics["backbone_layer_agg"] = str(args.backbone_layer_agg)
    metrics["backbone_layer_ids"] = str(getattr(args, "backbone_layer_ids", ""))
    metrics["backbone_ckpt"] = str(getattr(args, "backbone_ckpt", ""))

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json

        json.dump(metrics, f, indent=2)

    if bool(args.save_image_scores_csv) and image_score_rows is not None:
        scores_path = os.path.join(run_dir, str(args.image_scores_csv_name))
        with open(scores_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "is_anomaly", "image_score"])
            w.writeheader()
            for row in image_score_rows:
                w.writerow(
                    {
                        "image_path": row.get("image_path", ""),
                        "is_anomaly": int(bool(row.get("is_anomaly", 0))),
                        "image_score": float(row.get("image_score", 0.0)),
                    }
                )
        print(f"Wrote {scores_path}")

    print(f"Wrote {metrics_path}")
    return 0


def main() -> int:
    args = parse_args()
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
