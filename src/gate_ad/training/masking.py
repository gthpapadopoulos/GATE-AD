"""Random masking for patch graphs (coverage-aware)."""

from __future__ import annotations

import numpy as np
import torch


def build_neighbors(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build undirected neighbor lists from directed edge_index."""
    neigh = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [list(s) for s in neigh]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        if s == d:
            continue
        neigh[s].add(d)
        neigh[d].add(s)
    return [list(s) for s in neigh]


def random_mask(
    num_nodes: int,
    neighbors: list[list[int]],
    coverage: np.ndarray,
    mask_ratio: float,
    rng: np.random.RandomState,
    device: torch.device,
) -> torch.Tensor:
    """Random masking with coverage and neighbor-blocking constraints."""
    target = max(1, int(num_nodes * mask_ratio))
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)
    nodes.sort(key=lambda i: int(coverage[i]))

    selected = []
    blocked = set()
    for i in nodes:
        if len(selected) >= target:
            break
        if i in blocked:
            continue
        selected.append(i)
        blocked.add(i)
        for nb in neighbors[i]:
            blocked.add(nb)

    if len(selected) < target:
        missing = target - len(selected)
        uncovered = [i for i in range(num_nodes) if (not coverage[i]) and i not in selected]
        rng.shuffle(uncovered)
        take = min(missing, len(uncovered))
        selected.extend(uncovered[:take])
        missing -= take
        if missing > 0:
            remaining = [i for i in range(num_nodes) if i not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[:missing])

    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[selected] = True
    coverage[selected] = True
    return mask
