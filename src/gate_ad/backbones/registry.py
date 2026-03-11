"""Backbone registry and shared base wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from PIL import Image


class Backbone(Protocol):
    model_name: str
    device: str
    smaller_edge_size: int
    half_precision: bool

    def prepare_image(self, img: str | np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        ...

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class BackboneConfig:
    model_name: str
    device: str = "cuda:0"
    smaller_edge_size: int = 448
    half_precision: bool = False


class VisionTransformerWrapper:
    """Base wrapper for ViT-style backbones."""

    def __init__(self, model_name: str, device: str, smaller_edge_size: int = 448, half_precision: bool = False):
        self.model_name = str(model_name)
        self.device = str(device)
        self.smaller_edge_size = int(smaller_edge_size)
        self.half_precision = bool(half_precision)
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError

    def prepare_image(self, img: str | np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        raise NotImplementedError

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _infer_patch_size_from_name(model_name: str) -> int:
    name = model_name.lower()
    if "16" in name:
        return 16
    if "14" in name:
        return 14
    if "8" in name:
        return 8
    return 0


def adjust_edge_size_for_dinov3(model_name: str, smaller_edge_size: int) -> int:
    """Match DINOv2 token density when using DINOv3 at 448 default."""
    if model_name.startswith("dinov3") and int(smaller_edge_size) == 448:
        ps = _infer_patch_size_from_name(model_name)
        if ps:
            return ps * 32
    return int(smaller_edge_size)


def resolve_weights_dir(default_repo_root: str) -> str:
    return os.environ.get("GATEAD_WEIGHTS_DIR") or os.path.join(
        default_repo_root, "experiments", "exp_2025-11-03", "weights"
    )


def to_pil(img: str | np.ndarray) -> Image.Image:
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    raise TypeError(f"Unsupported image type: {type(img)}")
