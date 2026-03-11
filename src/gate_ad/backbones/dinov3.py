"""DINOv3 backbone wrapper."""

from __future__ import annotations

import os
import numpy as np
import torch
from torchvision import transforms

from .registry import VisionTransformerWrapper, resolve_weights_dir, to_pil


class DINOv3Wrapper(VisionTransformerWrapper):
    def load_model(self):

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        local_repo = os.path.join(repo_root, "dinov3")
        cache_repo = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")

        def _has_hubconf(path: str) -> bool:
            return os.path.isfile(os.path.join(path, "hubconf.py"))

        if _has_hubconf(local_repo):
            repo = local_repo
            source = "local"
        elif _has_hubconf(cache_repo):
            repo = cache_repo
            source = "local"
        else:
            repo = "facebookresearch/dinov3"
            source = "github"

        ckpt_path = os.environ.get("GATEAD_DINOV3_CKPT")
        if ckpt_path and not os.path.isfile(ckpt_path):
            print(f"Warning: GATEAD_DINOV3_CKPT set but file not found: {ckpt_path}")
            ckpt_path = None


        if ckpt_path is None:
            weights_dir = resolve_weights_dir(repo_root)
            if os.path.isdir(weights_dir):
                cand = [c for c in os.listdir(weights_dir) if c.endswith((".pth", ".pt"))]
                mn = self.model_name.lower()
                mn_short = mn.replace("dinov3_", "")
                sel = [c for c in cand if mn in c.lower() or mn_short in c.lower()]
                if not sel and "vit" in mn_short:
                    for token in ["vitg", "vitl", "vith", "vitb", "vits"]:
                        if token in mn_short:
                            sel = [c for c in cand if token in c.lower()]
                            if sel:
                                break
                if sel:
                    ckpt_path = os.path.join(weights_dir, sorted(sel)[0])

        try:
            if ckpt_path is not None:
                model = torch.hub.load(repo, self.model_name, source=source, weights=ckpt_path)
                print(f"Loaded DINOv3 weights from {ckpt_path}")
            else:
                model = torch.hub.load(repo, self.model_name, source=source)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 backbone {self.model_name} (repo={repo}, source={source}, ckpt={ckpt_path}): {e}"
            ) from e

        model.eval()


        self.patch_size = None
        if hasattr(model, "patch_size"):
            ps = getattr(model, "patch_size")
            if isinstance(ps, (tuple, list)):
                ps = ps[0]
            self.patch_size = int(ps)
        elif hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
            ps = model.patch_embed.patch_size
            if isinstance(ps, (tuple, list)):
                ps = ps[0]
            self.patch_size = int(ps)


        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if ckpt_path:
            w = str(ckpt_path).lower()
            if "sat493m" in w or w.endswith("-eadcf0ff.pth") or w.endswith("-eadcf0ff.pt"):
                mean = (0.430, 0.411, 0.296)
                std = (0.213, 0.156, 0.143)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=self.smaller_edge_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return model.to(self.device)

    def prepare_image(self, img: str | np.ndarray):
        image = to_pil(img)
        image_tensor = self.transform(image)
        height, width = image_tensor.shape[1:]
        ps = self.patch_size or getattr(self.model, "patch_size", 16)
        cropped_width, cropped_height = width - width % ps, height - height % ps
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        grid_size = (cropped_height // ps, cropped_width // ps)
        return image_tensor, grid_size

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)
            layer_ids_raw = str(os.environ.get("DINO_LAYER_IDS", "")).strip()
            n_sel: int | tuple[int, ...]
            if layer_ids_raw:
                try:
                    n_sel = tuple(int(tok.strip()) for tok in layer_ids_raw.split(",") if tok.strip())
                except Exception as e:
                    raise ValueError(f"Invalid DINO_LAYER_IDS='{layer_ids_raw}'. Expected comma-separated ints.") from e
                if len(n_sel) == 0:
                    raise ValueError("DINO_LAYER_IDS is set but empty after parsing.")
            else:
                n_sel = int(os.environ.get("DINO_LAST_N_LAYERS", 4))
            agg = str(os.environ.get("DINO_LAYER_AGG", "avg")).lower()
            outs = self.model.get_intermediate_layers(image_batch, n=n_sel, reshape=False, norm=True)
            if agg == "concat":
                tokens = torch.cat(tuple(outs), dim=-1)
            else:
                tokens = sum(o for o in outs) / float(len(outs))
            tokens = tokens.squeeze(0)
            tokens = torch.nn.functional.normalize(tokens, dim=-1)
        return tokens
