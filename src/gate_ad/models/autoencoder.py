"""Patch-graph autoencoder (GraphMAE-style)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .gnn import GATNeighborLayer


class PatchGraphAutoencoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        gnn_layers: int = 2,
        gnn_hidden_dims: Sequence[int] | None = None,
        gat_heads: int = 1,
        gat_self_loops: bool = False,
        latent_dim: int = 256,
        use_pred_head: bool = True,
        use_target_proj: bool = True,
        use_mlp: bool = True,
        use_residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.gnn_layers = int(gnn_layers)



        self.use_residual = bool(use_residual)
        self.gat_heads = int(gat_heads)
        self.gat_self_loops = bool(gat_self_loops)
        if self.gnn_layers < 1:
            raise ValueError("gnn_layers must be >= 1")

        if gnn_hidden_dims is None:

            gnn_hidden_dims = [int(latent_dim)] * self.gnn_layers
        else:
            gnn_hidden_dims = [int(d) for d in gnn_hidden_dims]
            if len(gnn_hidden_dims) != self.gnn_layers:
                raise ValueError("gnn_hidden_dims must have length == gnn_layers")
            if any(d <= 0 for d in gnn_hidden_dims):
                raise ValueError("gnn_hidden_dims must be positive")

        self.gnn_hidden_dims = list(gnn_hidden_dims)
        self.final_gnn_dim = int(self.gnn_hidden_dims[-1])
        self.use_pred_head = bool(use_pred_head)
        self.use_target_proj = bool(use_target_proj)
        if not self.use_pred_head and self.latent_dim != self.final_gnn_dim:
            raise ValueError("When use_pred_head=False, latent_dim must equal final GNN dim.")
        if not self.use_target_proj:
            pred_dim = self.final_gnn_dim if not self.use_pred_head else self.latent_dim
            if pred_dim != self.in_dim:
                raise ValueError("When use_target_proj=False, pred dim must equal in_dim.")

        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.gnn = nn.ModuleList()
        for li in range(self.gnn_layers):
            layer_in = in_dim if li == 0 else int(self.gnn_hidden_dims[li - 1])
            layer_out = int(self.gnn_hidden_dims[li])
            self.gnn.append(
                GATNeighborLayer(
                    layer_in,
                    layer_out,
                    dropout=self.dropout,
                    heads=self.gat_heads,
                    add_self_loops=self.gat_self_loops,
                )
            )
        self._inter_norms = nn.ModuleList(
            [nn.LayerNorm(int(self.gnn_hidden_dims[li])) for li in range(self.gnn_layers - 1)]
        )
        first_dim = int(self.gnn_hidden_dims[0]) if self.gnn_layers > 0 else int(in_dim)
        self._jk_proj = nn.Identity() if first_dim == self.final_gnn_dim else nn.Linear(first_dim, self.final_gnn_dim, bias=False)
        self._jk_out_norm = nn.LayerNorm(self.final_gnn_dim)

        if use_mlp:
            mlp_drop = nn.Identity() if self.dropout <= 0 else nn.Dropout(p=self.dropout)
            self.encoder = nn.Sequential(
                nn.Linear(self.final_gnn_dim, self.final_gnn_dim),
                nn.ReLU(inplace=True),
                mlp_drop,
                nn.Linear(self.final_gnn_dim, self.final_gnn_dim),
            )
        else:
            self.encoder = nn.Identity()
        self.decoder = nn.Sequential(
            nn.Linear(self.final_gnn_dim, self.final_gnn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_gnn_dim, in_dim),
        )

        self.to_latent_pred = nn.Identity() if not self.use_pred_head else nn.Linear(self.final_gnn_dim, latent_dim)
        self.to_latent_target = (
            nn.Identity() if not self.use_target_proj else nn.Linear(in_dim, latent_dim)
        )

    def _encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        h_first = None
        for li, layer in enumerate(self.gnn):
            out = layer(h, edge_index)
            if self.gnn_layers > 1 and li < self.gnn_layers - 1:
                out = self._inter_norms[li](out)
            if li == 0:
                h_first = out
            h = out
        if self.use_residual and self.gnn_layers > 1 and h_first is not None:

            h = self._jk_out_norm(h + self._jk_proj(h_first))
        return h

    def forward_latent(self, x: torch.Tensor, edge_index: torch.Tensor, *, target_x=None):
        h_mp = self._encode_graph(x, edge_index)
        h_enc = self.encoder(h_mp)
        z_pred = self.to_latent_pred(h_enc)
        z_tgt = self.to_latent_target(x if target_x is None else target_x)
        return z_pred, z_tgt

    def forward_input(self, x: torch.Tensor, edge_index: torch.Tensor, *, target_x=None):
        h_mp = self._encode_graph(x, edge_index)
        h_enc = self.encoder(h_mp)
        x_pred = self.decoder(h_enc)
        x_tgt = x if target_x is None else target_x
        return x_pred, x_tgt
