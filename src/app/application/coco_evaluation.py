"""Pure, testable helpers shared by the COCO engine-evaluation runner."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


def validate_coco_label_mapping(
    labels: Sequence[str], categories: Iterable[dict[str, Any]]
) -> dict[int, int]:
    """Map contiguous model class indexes to COCO's non-contiguous IDs.

    The model's label text must exactly match the category names in the supplied
    annotation file.  Failing closed avoids silently evaluating class indexes
    against wrong COCO categories.
    """
    category_by_name = {str(item["name"]): int(item["id"]) for item in categories}
    missing = [name for name in labels if name not in category_by_name]
    if missing:
        raise ValueError(f"model labels not present in COCO categories: {missing}")
    if len(set(labels)) != len(labels):
        raise ValueError("model labels must be unique")
    return {index: category_by_name[name] for index, name in enumerate(labels)}


def coco_box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """IoU for COCO-format ``[left, top, width, height]`` boxes."""
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[0]) + float(left[2]), float(right[0]) + float(right[2]))
    y2 = min(float(left[1]) + float(left[3]), float(right[1]) + float(right[3]))
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    left_area = max(float(left[2]), 0.0) * max(float(left[3]), 0.0)
    right_area = max(float(right[2]), 0.0) * max(float(right[3]), 0.0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def fixed_threshold_person_metrics(
    predictions: Iterable[dict[str, Any]],
    annotations: Iterable[dict[str, Any]],
    *,
    person_category_id: int,
    score_threshold: float,
    match_iou_threshold: float = 0.50,
) -> dict[str, float | int | None]:
    """Return simple non-crowd person P/R/F1 at a fixed score threshold.

    This intentionally complements, rather than replaces, COCOeval AP.  The
    matching is confidence-descending and one-to-one per image, using only
    non-crowd annotations.  It is an operating-point metric for deployment.
    """
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1")
    if not 0.0 <= match_iou_threshold <= 1.0:
        raise ValueError("match_iou_threshold must be between 0 and 1")
    ground_truth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if int(annotation.get("category_id", -1)) != person_category_id:
            continue
        if int(annotation.get("iscrowd", 0)):
            continue
        ground_truth[int(annotation["image_id"])].append(annotation)
    candidates = [
        item for item in predictions
        if int(item.get("category_id", -1)) == person_category_id
        and float(item.get("score", 0.0)) >= score_threshold
    ]
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    matched: dict[int, set[int]] = defaultdict(set)
    true_positive = false_positive = 0
    for prediction in candidates:
        image_id = int(prediction["image_id"])
        best_index = None
        best_iou = 0.0
        for index, annotation in enumerate(ground_truth.get(image_id, ())):
            if index in matched[image_id]:
                continue
            iou = coco_box_iou(prediction["bbox"], annotation["bbox"])
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_index is not None and best_iou >= match_iou_threshold:
            matched[image_id].add(best_index)
            true_positive += 1
        else:
            false_positive += 1
    false_negative = sum(len(items) for items in ground_truth.values()) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "score_threshold": score_threshold,
        "match_iou_threshold": match_iou_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "ground_truth": true_positive + false_negative,
        "predictions": len(candidates),
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
        "f1": round(f1, 6) if f1 is not None else None,
    }
