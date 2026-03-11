"""Run default dataset sweeps from YAML config."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml


FIXED_BORDER_PATCHES = 3


def _as_list(v):
    if isinstance(v, list):
        return v
    return [v]


def _default_model_for_dataset(dataset: str) -> str:
    ds = str(dataset).lower()
    if ds == "mvtec":
        return "dinov2_vitl14_reg"
    if ds == "visa":
        return "dinov3_vitb16"
    if ds == "mpdd":
        return "dinov3_vitl16"
    raise ValueError(f"Unsupported dataset={dataset!r}. Expected mvtec|visa|mpdd")


def _default_resolution_for_model(model_name: str) -> int:
    if str(model_name).lower().startswith("dinov2"):
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


def _aggregate_avg_row(
    out_dir: Path,
    dataset: str,
    model_name: str,
    shots: int,
    seed: int,
    topk_ratio: float,
    run_prefix: str,
):
    pattern = f"{run_prefix}_{dataset}_*_{model_name}_k{shots}_seed{seed}/metrics.json"
    paths = sorted(out_dir.glob(pattern))

    metrics = {
        "image_auroc": [],
        "image_ap": [],
        "image_f1_max": [],
        "pixel_auroc": [],
        "pixel_aupro": [],
        "train_feature_extract_time_per_image_sec": [],
        "train_total_time_per_image_sec": [],
        "train_wall_time_per_image_sec": [],
        "eval_avg_time_per_image_sec": [],
        "eval_avg_time_per_image_ms": [],
        "feature_extract_time_per_image_sec": [],
        "inference_time_per_image_sec": [],
    }

    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for k in metrics:
            v = data.get(k)
            if isinstance(v, (int, float)) and not math.isnan(v):
                metrics[k].append(float(v))

    row = {
        "dataset": dataset,
        "shots": int(shots),
        "seed": int(seed),
        "model": model_name,
        "topk_ratio": float(topk_ratio),
        "border_patches": int(FIXED_BORDER_PATCHES),
        "n_classes": len(paths),
    }
    for k, vals in metrics.items():
        row[k] = (sum(vals) / len(vals)) if vals else float("nan")

    csv_path = out_dir / f"avg_{dataset}.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Wrote avg row to {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run dataset defaults from YAML config.")
    p.add_argument("--config", required=True, help="Path to default YAML config.")
    p.add_argument("--out_dir", default="", help="Override output directory.")
    p.add_argument(
        "--enable_per_object_visualization",
        dest="enable_per_object_visualization",
        action="store_true",
        default=None,
    )
    p.add_argument(
        "--disable_per_object_visualization",
        dest="enable_per_object_visualization",
        action="store_false",
    )
    p.add_argument("--visualization_out_dir", type=str, default=None)
    p.add_argument("--visualization_max_per_type", type=int, default=None)
    p.add_argument(
        "--visualization_only_anomaly",
        dest="visualization_only_anomaly",
        action="store_true",
        default=None,
    )
    p.add_argument(
        "--visualization_all_samples",
        dest="visualization_only_anomaly",
        action="store_false",
    )
    p.add_argument("--visualization_overlay_alpha", type=float, default=None)
    p.add_argument(
        "--visualization_panel_layout",
        type=str,
        default=None,
        choices=["quad", "triptych", "separate"],
    )
    p.add_argument("--visualization_gaussian_sigma", type=float, default=None)
    p.add_argument("--visualization_colormap", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parents[2]

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = str(cfg["dataset"]).strip().lower()
    if dataset not in {"mvtec", "visa", "mpdd"}:
        raise ValueError(f"Unsupported dataset={dataset!r}. Expected mvtec|visa|mpdd")

    data_root = str(cfg["data_root"])
    if data_root.startswith("./"):
        data_root = str((repo_root / data_root[2:]).resolve())

    visa_split_csv = str(cfg.get("visa_split_csv", "")).strip()
    if visa_split_csv.startswith("./"):
        visa_split_csv = str((repo_root / visa_split_csv[2:]).resolve())

    model_name = str(cfg.get("model_name", _default_model_for_dataset(dataset)))
    device = str(cfg.get("device", "cuda:0"))
    resolution = int(cfg.get("resolution", _default_resolution_for_model(model_name)))
    backbone_last_n_layers = int(cfg.get("backbone_last_n_layers", 8))
    backbone_layer_agg = str(cfg.get("backbone_layer_agg", "avg"))
    backbone_ckpt = str(cfg.get("backbone_ckpt", "")).strip()
    if backbone_ckpt.startswith("./"):
        backbone_ckpt = str((repo_root / backbone_ckpt[2:]).resolve())

    epochs = int(cfg.get("epochs", 2000))
    lr = float(cfg.get("lr", 3e-4))
    mask_ratio = float(cfg.get("mask_ratio", 0.2))
    a = float(cfg.get("a", 2.0))

    gnn_layers = int(cfg.get("gnn_layers", 3))
    latent_dim = int(cfg.get("latent_dim", 256))
    gnn_hidden_dims_list = _as_list(cfg.get("gnn_hidden_dims", []))
    gnn_hidden_dims = ",".join(str(int(x)) for x in gnn_hidden_dims_list) if gnn_hidden_dims_list else ""
    dropout = float(cfg.get("dropout", 0.3))

    topk_ratio = float(cfg.get("topk_ratio", _default_topk_ratio_for_dataset(dataset)))
    image_score_pool = str(cfg.get("image_score_pool", _default_image_score_pool_for_dataset(dataset)))

    enable_per_object_visualization = bool(cfg.get("enable_per_object_visualization", False))
    if args.enable_per_object_visualization is not None:
        enable_per_object_visualization = bool(args.enable_per_object_visualization)

    visualization_out_dir = str(cfg.get("visualization_out_dir", "")).strip()
    if args.visualization_out_dir is not None:
        visualization_out_dir = str(args.visualization_out_dir).strip()
    if visualization_out_dir.startswith("./"):
        visualization_out_dir = str((repo_root / visualization_out_dir[2:]).resolve())

    visualization_max_per_type = int(cfg.get("visualization_max_per_type", 5))
    if args.visualization_max_per_type is not None:
        visualization_max_per_type = int(args.visualization_max_per_type)

    visualization_only_anomaly = bool(cfg.get("visualization_only_anomaly", False))
    if args.visualization_only_anomaly is not None:
        visualization_only_anomaly = bool(args.visualization_only_anomaly)

    visualization_overlay_alpha = float(cfg.get("visualization_overlay_alpha", 0.4))
    if args.visualization_overlay_alpha is not None:
        visualization_overlay_alpha = float(args.visualization_overlay_alpha)

    visualization_panel_layout = str(cfg.get("visualization_panel_layout", "quad"))
    if args.visualization_panel_layout is not None:
        visualization_panel_layout = str(args.visualization_panel_layout)

    visualization_gaussian_sigma = float(cfg.get("visualization_gaussian_sigma", 4.0))
    if args.visualization_gaussian_sigma is not None:
        visualization_gaussian_sigma = float(args.visualization_gaussian_sigma)

    visualization_colormap = str(cfg.get("visualization_colormap", "magma"))
    if args.visualization_colormap is not None:
        visualization_colormap = str(args.visualization_colormap)

    shots_list = [int(x) for x in _as_list(cfg.get("shots_list", [1]))]
    if "seeds_list" in cfg:
        seeds_list = [int(x) for x in _as_list(cfg.get("seeds_list", [0]))]
    else:
        seeds_list = [int(cfg.get("seed", 0))]

    objects = ",".join(str(x) for x in _as_list(cfg.get("objects", [])))

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = repo_root / f"outputs_{dataset}_default"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = None
    if model_name.startswith("dinov3"):

        env = None

    for seed in seeds_list:
        for shots in shots_list:
            run_prefix = f"{dataset}_k{shots}_seed{seed}_topk{topk_ratio}_b{FIXED_BORDER_PATCHES}"
            cmd = [
                sys.executable,
                "-m",
                "gate_ad.cli.run_sweep",
                "--dataset",
                dataset,
                "--data_root",
                data_root,
                "--model_name",
                model_name,
                "--device",
                device,
                "--resolution",
                str(resolution),
                "--backbone_last_n_layers",
                str(backbone_last_n_layers),
                "--backbone_layer_agg",
                backbone_layer_agg,
                "--shots",
                str(shots),
                "--seed",
                str(seed),
                "--epochs",
                str(epochs),
                "--lr",
                str(lr),
                "--mask_ratio",
                str(mask_ratio),
                "--a",
                str(a),
                "--gnn_layers",
                str(gnn_layers),
                "--latent_dim",
                str(latent_dim),
                "--dropout",
                str(dropout),
                "--topk_ratio",
                str(topk_ratio),
                "--image_score_pool",
                image_score_pool,
                "--out_dir",
                str(out_dir),
                "--run_prefix",
                run_prefix,
            ]
            if backbone_ckpt:
                cmd += ["--backbone_ckpt", backbone_ckpt]
            if gnn_hidden_dims:
                cmd += ["--gnn_hidden_dims", gnn_hidden_dims]
            if dataset == "visa":
                cmd += ["--visa_split_csv", visa_split_csv]
            if dataset == "mpdd" and objects:
                cmd += ["--objects", objects]
            if enable_per_object_visualization:
                cmd += ["--enable_per_object_visualization"]
                if visualization_out_dir:
                    cmd += ["--visualization_out_dir", visualization_out_dir]
                cmd += ["--visualization_max_per_type", str(visualization_max_per_type)]
                cmd += ["--visualization_overlay_alpha", str(visualization_overlay_alpha)]
                cmd += ["--visualization_panel_layout", str(visualization_panel_layout)]
                cmd += ["--visualization_gaussian_sigma", str(visualization_gaussian_sigma)]
                cmd += ["--visualization_colormap", str(visualization_colormap)]
                if visualization_only_anomaly:
                    cmd += ["--visualization_only_anomaly"]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env)

            _aggregate_avg_row(
                out_dir=out_dir,
                dataset=dataset,
                model_name=model_name,
                shots=shots,
                seed=seed,
                topk_ratio=topk_ratio,
                run_prefix=run_prefix,
            )

    print(f"Done. Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
