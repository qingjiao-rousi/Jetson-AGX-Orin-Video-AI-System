"""Pure helpers for one-to-one detector-output comparisons.

The helpers intentionally know nothing about DeepStream, tracking, routing, or
TensorRT.  This keeps an FP16/INT8 alignment report limited to raw detector
boxes after the same decoder/NMS has been applied to both engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.domain.entities import Detection


@dataclass(frozen=True)
class DetectionMatch:
    """One class-consistent, one-to-one match between two detector outputs."""

    reference_index: int
    candidate_index: int
    iou: float
    confidence_delta: float


def bbox_iou(left: Detection, right: Detection) -> float:
    """Return IoU for two detection boxes."""
    left_box, right_box = left.bbox, right.bbox
    x1 = max(left_box.left, right_box.left)
    y1 = max(left_box.top, right_box.top)
    x2 = min(left_box.left + left_box.width, right_box.left + right_box.width)
    y2 = min(left_box.top + left_box.height, right_box.top + right_box.height)
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    left_area = max(left_box.width, 0.0) * max(left_box.height, 0.0)
    right_area = max(right_box.width, 0.0) * max(right_box.height, 0.0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    reference: Sequence[Detection],
    candidate: Sequence[Detection],
    *,
    iou_threshold: float,
) -> tuple[tuple[DetectionMatch, ...], tuple[int, ...], tuple[int, ...]]:
    """Greedily match highest-IoU boxes of the same class exactly once.

    Returns ``(matches, reference_only_indices, candidate_only_indices)``.
    Greedy matching on the globally sorted candidate pairs avoids allowing one
    high-overlap box to be counted more than once.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    pairs: list[DetectionMatch] = []
    for reference_index, reference_detection in enumerate(reference):
        for candidate_index, candidate_detection in enumerate(candidate):
            if reference_detection.class_id != candidate_detection.class_id:
                continue
            iou = bbox_iou(reference_detection, candidate_detection)
            if iou >= iou_threshold:
                pairs.append(
                    DetectionMatch(
                        reference_index=reference_index,
                        candidate_index=candidate_index,
                        iou=iou,
                        confidence_delta=candidate_detection.confidence - reference_detection.confidence,
                    )
                )
    pairs.sort(key=lambda item: (-item.iou, item.reference_index, item.candidate_index))
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    matches: list[DetectionMatch] = []
    for pair in pairs:
        if pair.reference_index in used_reference or pair.candidate_index in used_candidate:
            continue
        used_reference.add(pair.reference_index)
        used_candidate.add(pair.candidate_index)
        matches.append(pair)
    return (
        tuple(matches),
        tuple(index for index in range(len(reference)) if index not in used_reference),
        tuple(index for index in range(len(candidate)) if index not in used_candidate),
    )
