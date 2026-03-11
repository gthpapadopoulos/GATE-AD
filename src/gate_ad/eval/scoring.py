"""Scoring utilities (test-time masking and single-pass)."""

from __future__ import annotations

import numpy as np
import torch

from gate_ad.training.loss import patch_reconstruction_error
from gate_ad.training.masking import build_neighbors, random_mask


def mean_topk(scores: torch.Tensor, ratio: float) -> float:
    flat = scores.reshape(-1)
    k = max(1, int(len(flat) * ratio))
    top = torch.topk(flat, k=k, largest=True).values
    return float(top.mean().item())


@torch.inference_mode()
def score_single_pass(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    a: float,
) -> torch.Tensor:
    """Single forward pass (no test-time masking)."""
    z_pred, z_tgt = model.forward_latent(x, edge_index)
    return patch_reconstruction_error(z_pred, z_tgt, a=a)


@torch.inference_mode()
def score_test_time_masking(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    mask_ratio: float,
    a: float,
    full_coverage: bool = True,
    full_coverage_cap: int = 10000,
    test_masks: int = 0,
) -> torch.Tensor:
    """
    Iterative masked scoring.
    If full_coverage=True and test_masks=0, mask until all nodes covered or cap reached.
    """
    device = x.device
    num_nodes = x.shape[0]
    neighbors = build_neighbors(edge_index.cpu(), num_nodes)
    coverage = np.zeros(num_nodes, dtype=np.bool_)
    rng = np.random.RandomState(0)

    acc = torch.zeros(num_nodes, device=device)
    counts = torch.zeros(num_nodes, device=device)


    z_pred_full, z_tgt_full = model.forward_latent(x, edge_index)
    full_err = patch_reconstruction_error(
        z_pred_full,
        z_tgt_full,
        a=a,
    )

    iters = 0
    while True:
        if test_masks > 0 and iters >= test_masks:
            break
        if test_masks == 0 and bool(coverage.all()):
            break
        if test_masks == 0 and full_coverage and iters >= int(full_coverage_cap):
            break

        mask = random_mask(
            num_nodes=num_nodes,
            neighbors=neighbors,
            coverage=coverage,
            mask_ratio=mask_ratio,
            rng=rng,
            device=device,
        )
        mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
        x_masked = x * (1.0 - mask_f) + model.mask_token.unsqueeze(0) * mask_f
        z_pred, z_tgt = model.forward_latent(x_masked, edge_index, target_x=x)
        per_node = patch_reconstruction_error(
            z_pred,
            z_tgt,
            a=a,
        )

        acc[mask] += per_node[mask]
        counts[mask] += 1.0
        iters += 1

    err = acc / counts.clamp_min(1.0)
    err = torch.where(counts > 0, err, full_err)
    return err
