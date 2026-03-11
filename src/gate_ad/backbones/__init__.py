from .dinov2 import DINOv2Wrapper
from .dinov3 import DINOv3Wrapper
from .registry import BackboneConfig, VisionTransformerWrapper, adjust_edge_size_for_dinov3


def get_backbone(cfg: BackboneConfig):
    model_name = str(cfg.model_name)
    smaller_edge_size = adjust_edge_size_for_dinov3(model_name, cfg.smaller_edge_size)
    if model_name.startswith("dinov2"):
        return DINOv2Wrapper(model_name, cfg.device, smaller_edge_size, cfg.half_precision)
    if model_name.startswith("dinov3"):
        return DINOv3Wrapper(model_name, cfg.device, smaller_edge_size, cfg.half_precision)
    raise ValueError(f"Unknown backbone: {model_name}")
