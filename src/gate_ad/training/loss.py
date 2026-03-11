"""Loss functions for reconstruction-based training and scoring."""

from __future__ import annotations

import torch


def _normalize_last_dim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def patch_reconstruction_error(
    z_pred: torch.Tensor,
    z_tgt: torch.Tensor,
    *,
    a: float = 2.0,
) -> torch.Tensor:
    """Per-node SCE reconstruction error: (1 - cos(pred, tgt))^a."""
    zp_n = _normalize_last_dim(z_pred)
    zt_n = _normalize_last_dim(z_tgt)
    cos = torch.sum(zp_n * zt_n, dim=-1).clamp(-1.0, 1.0)
    base = 1.0 - cos
    return base ** float(a)


def masked_reconstruction_loss(
    z_pred: torch.Tensor,
    z_tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    a: float = 2.0,
) -> torch.Tensor:
    """Mean reconstruction loss over masked nodes only."""
    zp = z_pred[mask]
    zt = z_tgt[mask]
    if zp.numel() == 0:
        return torch.zeros((), device=z_pred.device, dtype=z_pred.dtype)
    per_node = patch_reconstruction_error(zp, zt, a=a)
    return torch.mean(per_node)
