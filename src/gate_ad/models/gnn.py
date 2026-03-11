"""GNN neighbor aggregation layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATNeighborLayer(nn.Module):
    """Single-head GAT layer (PyG GATConv)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        heads: int = 1,
        *,
        add_self_loops: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.dropout_p = float(dropout)
        self.heads = int(heads)
        self.add_self_loops = bool(add_self_loops)

        self.conv = GATConv(
            in_channels=self.in_dim,
            out_channels=self.out_dim,
            heads=self.heads,
            concat=False,
            dropout=self.dropout_p,
            add_self_loops=self.add_self_loops,
        )
        self.out_act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros((x.shape[0], self.out_dim), device=x.device, dtype=x.dtype)
        out = self.conv(x, edge_index)
        return self.out_act(out)
