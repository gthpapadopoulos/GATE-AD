"""Run a dataset sweep for GATE-AD defaults."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys

import torch

from gate_ad.cli.run_one import default_rotate_aug_mode, run_single


MVTEC_OBJECTS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_OBJECTS = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

MPDD_OBJECTS = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run a dataset sweep.")
    p.add_argument("--dataset", required=True, choices=["mvtec", "visa", "mpdd"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--visa_split_csv", default="")
    p.add_argument("--objects", default="", help="Comma-separated object list (optional).")

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
    p.add_argument("--gat_self_loops", action="store_true", default=False)
    p.add_argument("--no_pred_head", action="store_true", default=False)
    p.add_argument("--no_target_proj", action="store_true", default=False)
    p.add_argument("--gnn_residual", action="store_true", default=False)

    p.add_argument("--image_score_pool", type=str, default="auto", choices=["auto", "topk_mean", "max"])

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
    p.add_argument("--run_prefix", default="")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--enable_per_object_visualization", action="store_true", default=False)
    p.add_argument("--visualization_out_dir", type=str, default="")
    p.add_argument("--visualization_max_per_type", type=int, default=5)
    p.add_argument("--visualization_only_anomaly", action="store_true", default=False)
    p.add_argument("--visualization_overlay_alpha", type=float, default=0.4)
    p.add_argument(
        "--visualization_panel_layout",
        type=str,
        default="quad",
        choices=["quad", "triptych", "separate"],
    )
    p.add_argument("--visualization_gaussian_sigma", type=float, default=4.0)
    p.add_argument("--visualization_colormap", type=str, default="magma")
    p.add_argument("--save_image_scores_csv", action="store_true", default=False)
    p.add_argument("--image_scores_csv_name", type=str, default="image_anomaly_scores.csv")
    return p.parse_args()


def _metrics_valid(
    path: str,
    expected_a: float = 2.0,
) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        if not ("image_auroc" in j and "pixel_auroc" in j):
            return False
        actual_loss = str(j.get("loss_type", "sce")).lower()
        if actual_loss != "sce":
            return False
        actual_a = float(j.get("a", 2.0))
        if abs(actual_a - float(expected_a)) > 1e-12:
            return False
        actual_sel = str(j.get("shot_selection", "first")).lower()
        if actual_sel != "first":
            return False
        return True
    except Exception:
        return False


def _default_objects(dataset: str) -> list[str]:
    ds = str(dataset).lower()
    if ds == "visa":
        return VISA_OBJECTS
    if ds == "mpdd":
        return MPDD_OBJECTS
    return MVTEC_OBJECTS


def _visualization_root(args: argparse.Namespace) -> str:
    out = str(getattr(args, "visualization_out_dir", "")).strip()
    if out:
        return out
    root = os.path.join(args.out_dir, "visualizations")
    run_prefix = str(getattr(args, "run_prefix", "")).strip()
    if run_prefix:
        root = os.path.join(root, run_prefix)
    return root


def _has_visualization_outputs(root: str, object_name: str) -> bool:
    obj_dir = os.path.join(root, object_name)
    if not os.path.isdir(obj_dir):
        return False
    for r, _dirs, files in os.walk(obj_dir):
        for fn in files:
            if fn.lower().endswith(".png"):
                return True
    return False


def _visualization_cmd(args: argparse.Namespace, object_name: str, rotate_mode: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "gate_ad.cli.visualize_triptychs",
        "--dataset",
        args.dataset,
        "--object_name",
        object_name,
        "--data_root",
        args.data_root,
        "--model_name",
        args.model_name,
        "--device",
        args.device,
        "--backbone_last_n_layers",
        str(args.backbone_last_n_layers),
        "--backbone_layer_agg",
        str(args.backbone_layer_agg),
        "--shots",
        str(args.shots),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--mask_ratio",
        str(args.mask_ratio),
        "--a",
        str(args.a),
        "--gnn_layers",
        str(args.gnn_layers),
        "--latent_dim",
        str(args.latent_dim),
        "--dropout",
        str(args.dropout),
        "--max_per_type",
        str(args.visualization_max_per_type),
        "--overlay_alpha",
        str(args.visualization_overlay_alpha),
        "--panel_layout",
        str(args.visualization_panel_layout),
        "--gaussian_sigma",
        str(args.visualization_gaussian_sigma),
        "--colormap",
        str(args.visualization_colormap),
        "--out_dir",
        _visualization_root(args),
    ]
    if args.visa_split_csv:
        cmd += ["--visa_split_csv", str(args.visa_split_csv)]
    if args.backbone_layer_ids:
        cmd += ["--backbone_layer_ids", str(args.backbone_layer_ids)]
    if args.resolution is not None:
        cmd += ["--resolution", str(args.resolution)]
    if args.topk_ratio is not None:
        cmd += ["--topk_ratio", str(args.topk_ratio)]
    if args.backbone_ckpt:
        cmd += ["--backbone_ckpt", str(args.backbone_ckpt)]
    if args.gnn_hidden_dims:
        cmd += ["--gnn_hidden_dims", str(args.gnn_hidden_dims)]
    if bool(args.gat_self_loops):
        cmd.append("--gat_self_loops")
    if bool(args.no_pred_head):
        cmd.append("--no_pred_head")
    if bool(args.no_target_proj):
        cmd.append("--no_target_proj")
    if bool(args.no_mlp):
        cmd.append("--no_mlp")
    if bool(args.gnn_residual):
        cmd.append("--gnn_residual")
    if rotate_mode == "all":
        cmd += ["--rotate_aug_mode", "all"]
    if bool(args.augment_hflip):
        cmd.append("--augment_hflip")
    if bool(args.augment_vflip):
        cmd.append("--augment_vflip")
    if bool(args.augment_random_rotation):
        cmd += ["--augment_random_rotation", "--augment_random_rotation_n", str(args.augment_random_rotation_n)]
    if bool(args.augment_color_jitter):
        cmd += ["--augment_color_jitter", "--augment_color_jitter_n", str(args.augment_color_jitter_n)]
    if (not bool(args.no_test_time_masking)) and bool(args.test_time_masking):
        cmd += [
            "--test_time_masking",
            "--test_mask_ratio",
            str(args.test_mask_ratio),
            "--test_full_coverage_cap",
            str(args.test_full_coverage_cap),
        ]
    if bool(args.visualization_only_anomaly):
        cmd.append("--only_anomaly")
    return cmd


def main() -> int:
    args = parse_args()

    if args.objects.strip():
        objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    else:
        objects = _default_objects(args.dataset)

    for obj in objects:
        rotate_mode = default_rotate_aug_mode(args.dataset, obj)
        run_name = f"{args.dataset}_{obj}_{args.model_name}_k{args.shots}_seed{args.seed}"
        if args.run_prefix:
            run_name = f"{args.run_prefix}_{run_name}"

        metrics_path = os.path.join(args.out_dir, run_name, "metrics.json")
        skip_run = args.skip_existing and os.path.isfile(metrics_path) and _metrics_valid(
            metrics_path,
            expected_a=args.a,
        )
        if skip_run:
            print(f"Skip {run_name} (metrics.json valid)")
        else:
            run_args = argparse.Namespace(
                dataset=args.dataset,
                object_name=obj,
                data_root=args.data_root,
                visa_split_csv=args.visa_split_csv,
                model_name=args.model_name,
                device=args.device,
                resolution=args.resolution,
                backbone_last_n_layers=args.backbone_last_n_layers,
                backbone_layer_agg=args.backbone_layer_agg,
                backbone_layer_ids=args.backbone_layer_ids,
                backbone_ckpt=args.backbone_ckpt,
                shots=args.shots,
                seed=args.seed,
                epochs=args.epochs,
                lr=args.lr,
                mask_ratio=args.mask_ratio,
                a=args.a,
                gnn_layers=args.gnn_layers,
                latent_dim=args.latent_dim,
                gnn_hidden_dims=args.gnn_hidden_dims,
                gat_self_loops=bool(args.gat_self_loops),
                no_pred_head=args.no_pred_head,
                no_target_proj=args.no_target_proj,
                no_mlp=args.no_mlp,
                gnn_residual=args.gnn_residual,
                dropout=args.dropout,
                image_score_pool=args.image_score_pool,
                rotate_aug_mode=rotate_mode,
                augment_hflip=args.augment_hflip,
                augment_vflip=args.augment_vflip,
                augment_random_rotation=args.augment_random_rotation,
                augment_random_rotation_n=args.augment_random_rotation_n,
                augment_color_jitter=args.augment_color_jitter,
                augment_color_jitter_n=args.augment_color_jitter_n,
                test_time_masking=args.test_time_masking,
                no_test_time_masking=args.no_test_time_masking,
                test_mask_ratio=args.test_mask_ratio,
                test_full_coverage_cap=args.test_full_coverage_cap,
                topk_ratio=args.topk_ratio,
                out_dir=args.out_dir,
                run_name=run_name,
                save_image_scores_csv=bool(args.save_image_scores_csv),
                image_scores_csv_name=args.image_scores_csv_name,
            )

            run_single(run_args)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.enable_per_object_visualization:
            vis_root = _visualization_root(args)
            if args.skip_existing and _has_visualization_outputs(vis_root, obj):
                print(f"Skip visualization {obj} (already exists)")
            else:
                cmd = _visualization_cmd(args, obj, rotate_mode)
                print("Visualization:", " ".join(cmd))
                subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
