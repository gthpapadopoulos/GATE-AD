"""Seeding utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int, device: str = "cuda:0") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
