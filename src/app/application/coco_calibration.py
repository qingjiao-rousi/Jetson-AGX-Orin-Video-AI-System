"""Deterministic COCO-train image selection for TensorRT calibration."""

from __future__ import annotations

import random
from typing import Any, Sequence


def select_calibration_images(
    images: Sequence[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Return a stable random sample sorted by COCO image ID.

    Calibration does not consume labels, so this deliberately samples image
    metadata only.  The caller must supply train-split metadata, never the
    validation annotations used for evaluation.
    """
    if count <= 0:
        raise ValueError("count must be greater than zero")
    candidates = sorted(images, key=lambda item: int(item["id"]))
    if len(candidates) < count:
        raise ValueError(f"requested {count} images, but only {len(candidates)} are available")
    selected = random.Random(seed).sample(candidates, count)
    return sorted(selected, key=lambda item: int(item["id"]))
