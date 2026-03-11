"""VisA dataset loader."""

from __future__ import annotations

import csv
import os
from typing import Iterable

from .common import TestRecord, select_k_shot


def _abs_path(root: str, maybe_rel: str | None) -> str | None:
    if maybe_rel is None:
        return None
    p = str(maybe_rel).strip()
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


def load_visa_split(visa_split_csv: str, data_root: str, object_name: str):
    if not visa_split_csv:
        raise ValueError("visa_split_csv is required for VisA")
    if not os.path.isfile(visa_split_csv):
        raise FileNotFoundError(f"VisA split CSV not found: {visa_split_csv}")

    obj = str(object_name).strip()
    train_normals: list[str] = []
    test_records: list[TestRecord] = []
    with open(visa_split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if str(row.get("object", "")).strip() != obj:
                continue
            split = str(row.get("split", "")).strip().lower()
            label = str(row.get("label", "")).strip().lower()
            img_rel = str(row.get("image", "")).strip()
            mask_rel = str(row.get("mask", "")).strip()
            img_path = _abs_path(data_root, img_rel)
            if img_path is None:
                continue

            if split == "train" and label == "normal":
                train_normals.append(img_path)
            elif split == "test":
                is_anomaly = label == "anomaly"
                mask_path = _abs_path(data_root, mask_rel) if is_anomaly else None
                test_records.append(TestRecord(img_path, is_anomaly, mask_path))

    if not train_normals:
        raise RuntimeError(f"No VisA train/normal images found for object={obj} in {visa_split_csv}")
    if not test_records:
        raise RuntimeError(f"No VisA test records found for object={obj} in {visa_split_csv}")
    return train_normals, test_records


def get_train_normals(
    visa_split_csv: str,
    data_root: str,
    object_name: str,
    shots: int,
    seed: int,
    selection: str = "first",
    shot_block_size: int = 8,
):
    train_normals, _ = load_visa_split(visa_split_csv, data_root, object_name)
    return select_k_shot(train_normals, shots, seed, selection, block_size=shot_block_size)
