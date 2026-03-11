"""I/O helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


def read_image_rgb(path: str) -> Optional[np.ndarray]:
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def read_mask_gray(path: str) -> Optional[np.ndarray]:
    try:
        return np.asarray(Image.open(path).convert("L"))
    except Exception:
        return None
