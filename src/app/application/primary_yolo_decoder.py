"""Decoder for the project's exported primary YOLOv8 TensorRT output.

``export_yolov8_ds/export_yoloV8.py`` appends ``DeepStreamOutput`` to the
Ultralytics model.  Its exact output contract is ``[x1, y1, x2, y2, score,
class_id]`` with shape ``[batch, anchors, 6]``.  It has reduced classes but
has not applied NMS.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from app.application.helmet_service import bbox_iou
from app.domain.entities import BoundingBox, Detection


def decode_primary_yolov8_output(
    output: np.ndarray,
    *,
    image_shape: tuple[int, int],
    labels: Sequence[str],
    confidence_threshold: float,
    nms_iou_threshold: float,
) -> tuple[Detection, ...]:
    """Decode primary ``[B, N, 6]`` output into source-image detections."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be between 0 and 1")
    raw = np.asarray(output)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError(
            "primary YOLO output must have shape [1, anchors, 6] or [anchors, 6] "
            f"([x1, y1, x2, y2, score, class_id]), got {raw.shape}"
        )
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width")
    ratio = min(640.0 / width, 640.0 / height)
    pad_x = (640.0 - width * ratio) / 2.0
    pad_y = (640.0 - height * ratio) / 2.0
    by_class: dict[int, list[Detection]] = defaultdict(list)
    for x1, y1, x2, y2, confidence, class_value in raw:
        confidence = float(confidence)
        if not np.isfinite(confidence) or confidence < confidence_threshold or confidence > 1.0:
            continue
        class_id = int(round(float(class_value)))
        if not 0 <= class_id < len(labels):
            continue
        left = min(max((float(x1) - pad_x) / ratio, 0.0), float(width))
        top = min(max((float(y1) - pad_y) / ratio, 0.0), float(height))
        right = min(max((float(x2) - pad_x) / ratio, 0.0), float(width))
        bottom = min(max((float(y2) - pad_y) / ratio, 0.0), float(height))
        if right <= left or bottom <= top:
            continue
        by_class[class_id].append(
            Detection(class_id, str(labels[class_id]), confidence, BoundingBox(left, top, right - left, bottom - top))
        )
    kept: list[Detection] = []
    for candidates in by_class.values():
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        while candidates:
            selected = candidates.pop(0)
            kept.append(selected)
            candidates = [item for item in candidates if bbox_iou(selected.bbox, item.bbox) < nms_iou_threshold]
    return tuple(sorted(kept, key=lambda item: item.confidence, reverse=True))
