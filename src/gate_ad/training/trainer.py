"""Training loop for graph autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch

from .loss import masked_reconstruction_loss
from .masking import build_neighbors, random_mask


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 3e-4
    mask_ratio: float = 0.25
    a: float = 2.0
    keep_best: bool = True
    early_stop_loss_threshold: float = 5e-7
    device: str = "cuda:0"


def apply_input_mask(x: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    token = mask_token.to(dtype=x.dtype).unsqueeze(0)
    return x * (1.0 - m) + token * m


def train_graphs(
    model,
    graphs: Iterable,
    cfg: TrainConfig,
):
    """
    Train on an iterable of PyG Data graphs (single-image graphs).
    Each Data must have fields: x, edge_index.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    graphs_list = list(graphs)
    if not graphs_list:
        raise ValueError("No training graphs provided.")

    neighbors_per_graph = [build_neighbors(g.edge_index.cpu(), g.x.shape[0]) for g in graphs_list]
    coverage_per_graph = [np.zeros(g.x.shape[0], dtype=np.bool_) for g in graphs_list]
    rng = np.random.RandomState(0)

    best_loss = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        losses: List[float] = []
        for gi, g in enumerate(graphs_list):
            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            mask = random_mask(
                num_nodes=x.shape[0],
                neighbors=neighbors_per_graph[gi],
                coverage=coverage_per_graph[gi],
                mask_ratio=cfg.mask_ratio,
                rng=rng,
                device=device,
            )

            x_masked = apply_input_mask(x, mask, model.mask_token)
            z_pred, z_tgt = model.forward_latent(x_masked, edge_index, target_x=x)
            loss = masked_reconstruction_loss(
                z_pred,
                z_tgt,
                mask,
                a=cfg.a,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs} loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if cfg.keep_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if cfg.early_stop_loss_threshold is not None and epoch_loss <= float(cfg.early_stop_loss_threshold):
            print(f"Early stop: loss {epoch_loss:.6f} <= {cfg.early_stop_loss_threshold:.2e}")
            break

    if cfg.keep_best and best_state is not None:
        model.load_state_dict(best_state)
    return model
