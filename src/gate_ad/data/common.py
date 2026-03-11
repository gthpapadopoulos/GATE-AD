"""Shared dataset helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: str) -> list[str]:
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMG_EXTS)]


def select_k_shot(
    paths: Iterable[str],
    shots: int,
    seed: int,
    selection: str = "first",
    block_size: int = 8,
) -> list[str]:
    files = sorted(list(paths))
    if shots <= 0:
        raise ValueError("shots must be >= 1")
    if shots >= len(files):
        return files
    selection = str(selection).lower()
    if selection == "first":
        return files[:shots]
    if selection == "shift":


        n = len(files)
        start = (2 * int(seed)) % n
        end = start + int(shots)
        if end <= n:
            return files[start:end]
        return files[start:] + files[: end - n]
    if selection == "block":
        block = int(block_size)
        if block <= 0:
            raise ValueError("block_size must be >= 1 for block shot selection")
        if shots > block:
            raise ValueError(f"shots ({shots}) cannot exceed block_size ({block}) when selection='block'")
        n = len(files)
        start = (int(seed) * block) % n
        block_files = [files[(start + i) % n] for i in range(block)]
        return block_files[:shots]
    raise ValueError(f"Unsupported shot selection: {selection}. Use one of: first, shift, block")


@dataclass(frozen=True)
class TestRecord:
    image_path: str
    is_anomaly: bool
    mask_path: str | None
