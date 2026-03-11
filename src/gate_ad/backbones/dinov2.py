"""DINOv2 backbone wrapper."""

from __future__ import annotations

import os
import numpy as np
import torch
from torchvision import transforms

from .registry import VisionTransformerWrapper, to_pil


class DINOv2Wrapper(VisionTransformerWrapper):
    def load_model(self):

        local_repo = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        if os.path.isdir(local_repo):
            model = torch.hub.load(local_repo, self.model_name, source="local")
        else:
            model = torch.hub.load("facebookresearch/dinov2", self.model_name)


        ckpt_path = os.environ.get("GATEAD_DINOV2_CKPT")
        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            if isinstance(state, dict):
                cleaned = {}
                for k, v in state.items():
                    nk = k
                    if nk.startswith("module."):
                        nk = nk[len("module.") :]
                    if nk.startswith("backbone."):
                        nk = nk[len("backbone.") :]
                    cleaned[nk] = v
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                print(f"Loaded DINOv2 weights from {ckpt_path}")
                if missing:
                    print(f"[DINOv2] Missing keys: {len(missing)}")
                if unexpected:
                    print(f"[DINOv2] Unexpected keys: {len(unexpected)}")

        model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=self.smaller_edge_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return model.to(self.device)

    def prepare_image(self, img: str | np.ndarray):
        image = to_pil(img)
        image_tensor = self.transform(image)


        height, width = image_tensor.shape[1:]
        ps = int(self.model.patch_size)
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
