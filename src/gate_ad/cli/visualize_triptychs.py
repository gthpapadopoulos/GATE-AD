"""Generate per-object visualization panels for test images."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from gate_ad.backbones import BackboneConfig, get_backbone
from gate_ad.cli.run_one import default_rotate_aug_mode
from gate_ad.data.mvtec import get_test_records as mvtec_test_records, get_train_normals as mvtec_train_normals
from gate_ad.data.transforms import color_jitter_aug, hflip_image, random_rotate_aug, rotate_aug, vflip_image
from gate_ad.data.visa import get_train_normals as visa_train_normals, load_visa_split
from gate_ad.eval.scoring import score_single_pass, score_test_time_masking
from gate_ad.graph.builder import build_grid_graph
from gate_ad.models.autoencoder import PatchGraphAutoencoder
from gate_ad.training.trainer import TrainConfig, train_graphs
from gate_ad.utils.io import read_image_rgb, read_mask_gray
from gate_ad.utils.seed import set_all_seeds


FIXED_GRID_K = 8
FIXED_GAT_HEADS = 1


def parse_args():
    p = argparse.ArgumentParser(description="Generate per-object visualization panels.")
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
    p.add_argument("--gat_self_loops", action="store_true", default=False)
    p.add_argument("--no_pred_head", action="store_true", default=False)
    p.add_argument("--no_target_proj", action="store_true", default=False)
    p.add_argument("--gnn_residual", action="store_true", default=False)

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
    p.add_argument("--topk_ratio", type=float, default=None)

    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_per_type", type=int, default=5)
    p.add_argument("--only_anomaly", action="store_true", default=False)
    p.add_argument("--overlay_alpha", type=float, default=0.4)
    p.add_argument("--colormap", type=str, default="magma")
    p.add_argument("--gaussian_sigma", type=float, default=4.0)
    p.add_argument(
        "--panel_layout",
        type=str,
        default="quad",
        choices=["quad", "triptych", "separate"],
    )
    return p.parse_args()


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


def _set_backbone_env(last_n: int, agg: str, layer_ids: str = ""):
    os.environ["DINO_LAST_N_LAYERS"] = str(int(last_n))
    os.environ["DINO_LAYER_AGG"] = str(agg)
    layer_ids = str(layer_ids).strip()
    if layer_ids:
        os.environ["DINO_LAYER_IDS"] = layer_ids
    else:
        os.environ.pop("DINO_LAYER_IDS", None)


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


def _make_graphs_from_images(backbone, images):
    graphs = []
    for img in images:
        img_tensor, grid_size = backbone.prepare_image(img)
        feats = backbone.extract_features(img_tensor)
        feats = feats.clone().detach()
        g, _ = build_grid_graph(feats, grid_size, border_patches=0, grid_k=FIXED_GRID_K)
        graphs.append(g)
    return graphs


def _resize_bilinear(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    t = torch.nn.functional.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy()


def _resize_nearest(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    t = torch.nn.functional.interpolate(t, size=out_hw, mode="nearest")
    return t.squeeze(0).squeeze(0).cpu().numpy()


def _make_quad_panel(
    normal_ref_img: np.ndarray,
    test_img: np.ndarray,
    overlay_rgb: np.ndarray,
    gt_mask_bw: np.ndarray,
    out_path: str,
    title: str,
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(normal_ref_img)
    axes[0].set_title("Normal ref")
    axes[1].imshow(test_img)
    axes[1].set_title("Test image")
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Recon error overlay")
    axes[3].imshow(gt_mask_bw, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("GT mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_triptych(
    img: np.ndarray,
    score_map: np.ndarray,
    overlay_rgb: np.ndarray,
    out_path: str,
    title: str,
    colormap: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    im = axes[1].imshow(score_map, cmap=colormap)
    axes[1].set_title("Recon error")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_separate(
    normal_ref_img: np.ndarray,
    test_img: np.ndarray,
    overlay_rgb: np.ndarray,
    gt_mask_bw: np.ndarray,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    plt.imsave(os.path.join(out_dir, "normal_ref.png"), normal_ref_img.astype(np.uint8))
    plt.imsave(os.path.join(out_dir, "test_image.png"), test_img.astype(np.uint8))
    plt.imsave(os.path.join(out_dir, "overlay.png"), overlay_rgb.astype(np.uint8))
    plt.imsave(os.path.join(out_dir, "gt_mask.png"), gt_mask_bw.astype(np.uint8), cmap="gray", vmin=0, vmax=1)


def _anomaly_type(image_path: str, data_root: str, is_anomaly: bool) -> str:
    try:
        rel = os.path.relpath(image_path, data_root)
    except Exception:
        rel = image_path
    parts = rel.split(os.sep)
    if "test" in parts:
        i = parts.index("test")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "anomaly" if bool(is_anomaly) else "good"


def main() -> int:
    args = parse_args()
    if args.resolution is None:
        args.resolution = _default_resolution_for_model(args.model_name)
    if args.topk_ratio is None:
        args.topk_ratio = _default_topk_ratio_for_dataset(args.dataset)
    rotate_mode = str(args.rotate_aug_mode).strip().lower()
    if rotate_mode in {"", "auto"}:
        args.rotate_aug_mode = default_rotate_aug_mode(args.dataset, args.object_name)
    else:
        args.rotate_aug_mode = rotate_mode

    set_all_seeds(args.seed, args.device)
    _set_backbone_env(args.backbone_last_n_layers, args.backbone_layer_agg, args.backbone_layer_ids)

    backbone_ckpt = str(getattr(args, "backbone_ckpt", "")).strip()
    if backbone_ckpt:
        backbone_ckpt = os.path.abspath(backbone_ckpt)
        if not os.path.isfile(backbone_ckpt):
            raise FileNotFoundError(f"backbone_ckpt not found: {backbone_ckpt}")
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
    normal_ref_img = read_image_rgb(train_paths[0]) if train_paths else None

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

    train_graphs_list = _make_graphs_from_images(backbone, train_images)
    if not train_graphs_list:
        raise RuntimeError("No train graphs were created.")

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
    model = train_graphs(model, train_graphs_list, train_cfg)
    model = model.to(args.device)
    model.eval()

    out_root = os.path.join(args.out_dir, args.object_name)
    os.makedirs(out_root, exist_ok=True)
    seen_per_type = defaultdict(int)
    test_records = _load_test_records(args)

    for rec in test_records:
        if args.only_anomaly and not rec.is_anomaly:
            continue
        anomaly_type = _anomaly_type(rec.image_path, args.data_root, bool(rec.is_anomaly))
        if seen_per_type[anomaly_type] >= int(args.max_per_type):
            continue

        img = read_image_rgb(rec.image_path)
        if img is None:
            continue
        if normal_ref_img is None:
            normal_ref_img = img

        img_h, img_w = img.shape[:2]
        img_tensor, grid_size = backbone.prepare_image(img)
        feats = backbone.extract_features(img_tensor)
        g, grid_size = build_grid_graph(feats, grid_size, border_patches=0, grid_k=FIXED_GRID_K)

        x = g.x.to(args.device)
        edge_index = g.edge_index.to(args.device)
        if (not args.no_test_time_masking) and args.test_time_masking:
            err = score_test_time_masking(
                model,
                x,
                edge_index,
                mask_ratio=args.test_mask_ratio,
                a=args.a,
                full_coverage=True,
                full_coverage_cap=args.test_full_coverage_cap,
                test_masks=0,
            )
        else:
            err = score_single_pass(model, x, edge_index, args.a)

        err = err.detach().cpu().numpy()
        hp, wp = grid_size
        patch_grid = err.reshape(hp, wp).astype(np.float32)
        patch_grid = np.nan_to_num(patch_grid, nan=0.0, posinf=0.0, neginf=0.0)

        score_map = _resize_bilinear(patch_grid, (img_h, img_w))
        score_map = np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)
        if float(args.gaussian_sigma) > 0.0:
            score_map = gaussian_filter(score_map, sigma=float(args.gaussian_sigma))
            score_map = np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)

        mn = float(score_map.min())
        mx = float(score_map.max())
        if mx > mn:
            score_norm = (score_map - mn) / (mx - mn)
        else:
            score_norm = np.zeros_like(score_map)

        cmap = plt.get_cmap(str(args.colormap))
        heat = (cmap(score_norm)[:, :, :3] * 255.0).astype(np.uint8)
        overlay = ((1.0 - float(args.overlay_alpha)) * img + float(args.overlay_alpha) * heat).astype(np.uint8)

        gt_bw = np.zeros((img_h, img_w), dtype=np.uint8)
        if rec.is_anomaly and rec.mask_path:
            gt = read_mask_gray(rec.mask_path)
            if gt is not None:
                if gt.shape != (img_h, img_w):
                    gt = _resize_nearest(gt.astype(np.float32), (img_h, img_w))
                gt_bw = (gt > 0).astype(np.uint8)

        out_dir = os.path.join(out_root, anomaly_type)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(rec.image_path))[0]
        title = f"{args.object_name}/{anomaly_type} | {base}"

        if args.panel_layout == "quad":
            out_path = os.path.join(out_dir, f"{base}_quad.png")
            _make_quad_panel(normal_ref_img, img, overlay, gt_bw, out_path, title)
        elif args.panel_layout == "triptych":
            out_path = os.path.join(out_dir, f"{base}_triptych.png")
            _make_triptych(img, score_map, overlay, out_path, title, str(args.colormap))
        else:
            out_path = os.path.join(out_dir, f"{base}_separate")
            _save_separate(normal_ref_img, img, overlay, gt_bw, out_path)

        seen_per_type[anomaly_type] += 1

    print(f"Wrote visualizations under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
