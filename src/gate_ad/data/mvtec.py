"""MVTec dataset loader."""

from __future__ import annotations

import os
from typing import Iterable

from .common import TestRecord, list_images, select_k_shot


def get_train_normals(
    data_root: str,
    object_name: str,
    shots: int,
    seed: int,
    selection: str = "first",
    shot_block_size: int = 8,
) -> list[str]:
    train_good_dir = os.path.join(data_root, object_name, "train", "good")
    files = list_images(train_good_dir)
    selected = select_k_shot(files, shots, seed, selection, block_size=shot_block_size)
    return [os.path.join(train_good_dir, f) for f in selected]


def get_test_records(data_root: str, object_name: str) -> list[TestRecord]:
    test_root = os.path.join(data_root, object_name, "test")
    records: list[TestRecord] = []
    for anomaly_type in sorted(os.listdir(test_root)):
        type_dir = os.path.join(test_root, anomaly_type)
        if not os.path.isdir(type_dir):
            continue
        is_good = anomaly_type == "good"
        for name in list_images(type_dir):
            img_path = os.path.join(type_dir, name)
            if is_good:
                records.append(TestRecord(img_path, False, None))
            else:
                base_no_ext = os.path.splitext(name)[0]
                mask_path = os.path.join(
                    data_root,
                    object_name,
                    "ground_truth",
                    anomaly_type,
                    base_no_ext + "_mask.png",
                )
                records.append(TestRecord(img_path, True, mask_path))
    if not records:
        raise RuntimeError(f"No MVTec test records found for {object_name} at {test_root}")
    return records
