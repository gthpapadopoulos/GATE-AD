"""Graph builder for grid-only patch graphs (PyG Data)."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch_geometric.data import Data


def _crop_grid(
    feats: torch.Tensor, grid_size: tuple[int, int], border_patches: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Crop border patches from a flattened grid."""
    hp, wp = int(grid_size[0]), int(grid_size[1])
    b = int(border_patches)
    if b <= 0:
        return feats, (hp, wp)
    if 2 * b >= hp or 2 * b >= wp:
        raise ValueError(f"border_patches={b} too large for grid_size={grid_size}")

    feats_2d = feats.view(hp, wp, -1)
    feats_2d = feats_2d[b : hp - b, b : wp - b, :]
    new_size = (hp - 2 * b, wp - 2 * b)
    return feats_2d.contiguous().view(-1, feats.shape[-1]), new_size


def _neighbors_4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield (r - 1, c)
    if r + 1 < h:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < w:
        yield (r, c + 1)


def _neighbors_8(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield (rr, cc)


def _build_grid_edges(grid_size: tuple[int, int], grid_k: int) -> torch.Tensor:
    """
    Return edge_index for a grid graph, directed (both ways).

    - For grid_k in {4, 8}: use 4- or 8-neighborhood.
    - For other grid_k >= 1: connect each node to its grid_k nearest spatial neighbors
      by Euclidean distance on the (r, c) patch grid.
    """
    h, w = int(grid_size[0]), int(grid_size[1])
    k = int(grid_k)
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    if k in (4, 8):
        pass
    else:
        n = int(h * w)
        if n <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        if k >= n:
            raise ValueError(f"grid_k={k} too large for grid_size={grid_size} (max {n-1})")

    edges = []
    if k in (4, 8):
        for r in range(h):
            for c in range(w):
                src = r * w + c
                neigh = _neighbors_8(r, c, h, w) if k == 8 else _neighbors_4(r, c, h, w)
                for rr, cc in neigh:
                    dst = rr * w + cc
                    edges.append((src, dst))
    else:


        offsets: list[tuple[int, int, int]] = []
        for dr in range(-(h - 1), h):
            for dc in range(-(w - 1), w):
                if dr == 0 and dc == 0:
                    continue
                d2 = int(dr * dr + dc * dc)
                offsets.append((d2, dr, dc))
        offsets.sort()

        for r in range(h):
            for c in range(w):
                src = r * w + c
                cnt = 0
                for _d2, dr, dc in offsets:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        edges.append((src, rr * w + cc))
                        cnt += 1
                        if cnt >= k:
                            break

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_grid_graph(
    feats: torch.Tensor,
    grid_size: tuple[int, int],
    *,
    border_patches: int = 0,
    grid_k: int = 8,
) -> tuple[Data, tuple[int, int]]:
    """
    Build a grid-only patch graph from backbone tokens.

    Args:
      feats: (N, D) token features on device.
      grid_size: (H, W) patch grid size.
      border_patches: crop this many patch cells from each border.
      grid_k: neighborhood size. Use 4/8 for standard grid, or any k>=1 for k-nearest spatial neighbors.
    """
    if feats.ndim != 2:
        raise ValueError(f"feats must be 2D [N, D], got {tuple(feats.shape)}")

    feats, grid_size = _crop_grid(feats, grid_size, border_patches)
    h, w = int(grid_size[0]), int(grid_size[1])
    expected = h * w
    if feats.shape[0] != expected:
        raise ValueError(f"feats rows {feats.shape[0]} != H*W {expected}")

    edge_index = _build_grid_edges(grid_size, grid_k).to(feats.device)

    data = Data(x=feats, edge_index=edge_index)
    return data, grid_size
