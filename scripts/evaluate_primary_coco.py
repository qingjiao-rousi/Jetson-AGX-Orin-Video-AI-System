#!/usr/bin/env python3
"""Evaluate primary YOLO FP16 and INT8 TensorRT engines on COCO val2017."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.application.coco_evaluation import fixed_threshold_person_metrics, validate_coco_label_mapping
from app.application.helmet_service import TensorRTHelmetBackend, letterbox, load_labels
from app.domain.entities import Detection
from app.application.primary_yolo_decoder import decode_primary_yolov8_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True, help="COCO val2017 image directory.")
    parser.add_argument("--annotations", type=Path, required=True, help="instances_val2017.json path.")
    parser.add_argument("--fp16-engine", type=Path, default=Path("models/fp16/yolov8s.engine"))
    parser.add_argument("--int8-engine", type=Path, default=Path("models/int8/yolov8s_int8.engine"))
    parser.add_argument("--labels", type=Path, default=Path("models/labels.txt"))
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates all val2017 images; a positive value is a deterministic smoke subset.")
    parser.add_argument("--evaluation-confidence", type=float, default=0.001, help="Low score floor for COCO AP prediction files.")
    parser.add_argument("--fp16-operating-threshold", type=float, default=0.25)
    parser.add_argument("--int8-thresholds", default="0.25,0.20,0.15,0.10")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=300, help="Global post-NMS cap, matching the DeepStream config.")
    parser.add_argument("--output", type=Path, help="Output directory; defaults under outputs/coco_eval.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--int8-thresholds must be comma-separated numbers") from exc
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError("--int8-thresholds must contain values between 0 and 1")
    return thresholds


def prepare(image: np.ndarray) -> np.ndarray:
    model_image, _, _, _ = letterbox(image, 640, 640)
    rgb = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])


def decode(output: np.ndarray, image_shape: tuple[int, int], labels: tuple[str, ...], confidence: float, nms_iou: float, top_k: int) -> list[Detection]:
    detections = list(decode_primary_yolov8_output(
        output, image_shape=image_shape, labels=labels,
        confidence_threshold=confidence, nms_iou_threshold=nms_iou,
    ))
    return sorted(detections, key=lambda item: item.confidence, reverse=True)[:top_k]


def coco_predictions(image_id: int, detections: list[Detection], class_to_category: dict[int, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in detections:
        category_id = class_to_category.get(item.class_id)
        if category_id is None:
            continue
        rows.append({
            "image_id": image_id, "category_id": category_id,
            "bbox": [round(item.bbox.left, 3), round(item.bbox.top, 3), round(item.bbox.width, 3), round(item.bbox.height, 3)],
            "score": round(item.confidence, 8),
        })
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coco_metrics(coco_gt: Any, predictions: list[dict[str, Any]], image_ids: list[int], category_ids: list[int]) -> dict[str, float | None]:
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        return {key: None for key in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")}
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
    return {name: round(float(value), 6) if value >= 0 else None for name, value in zip(names, evaluator.stats[:6])}


def main() -> int:
    args = parse_args()
    if args.max_images < 0 or args.top_k <= 0:
        raise SystemExit("--max-images must be non-negative and --top-k must be positive")
    if not 0.0 <= args.evaluation_confidence <= 1.0 or not 0.0 <= args.fp16_operating_threshold <= 1.0 or not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise SystemExit("all thresholds must be between 0 and 1")
    int8_thresholds = parse_thresholds(args.int8_thresholds)
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise SystemExit("pycocotools is required; install: python3 -m pip install -r requirements-eval.txt") from exc
    images_dir, annotation_path, fp16_engine, int8_engine, labels_path = (
        resolve(args.images_dir), resolve(args.annotations), resolve(args.fp16_engine), resolve(args.int8_engine), resolve(args.labels)
    )
    for path, label in ((images_dir, "images directory"), (annotation_path, "annotations"), (fp16_engine, "FP16 engine"), (int8_engine, "INT8 engine"), (labels_path, "labels")):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")
    labels = load_labels(str(labels_path))
    coco_gt = COCO(str(annotation_path))
    class_to_category = validate_coco_label_mapping(labels, coco_gt.dataset.get("categories", ()))
    person_category_id = class_to_category[0]
    images = sorted(coco_gt.dataset.get("images", ()), key=lambda item: int(item["id"]))
    if args.max_images:
        images = images[:args.max_images]
    if not images:
        raise SystemExit("annotation file contains no images")
    image_ids = [int(item["id"]) for item in images]
    category_ids = sorted(class_to_category.values())
    annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_ids))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve(args.output) if args.output else PROJECT_ROOT / "outputs" / "coco_eval" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    fp16_backend, int8_backend = TensorRTHelmetBackend(str(fp16_engine)), TensorRTHelmetBackend(str(int8_engine))
    fp16_all: list[dict[str, Any]] = []
    int8_all: list[dict[str, Any]] = []
    fp16_operating: list[dict[str, Any]] = []
    int8_operating: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in int8_thresholds}
    for index, image_info in enumerate(images, start=1):
        image_path = images_dir / str(image_info["file_name"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read COCO image: {image_path}")
        tensor = prepare(image)
        fp16_raw, int8_raw = fp16_backend.infer(tensor), int8_backend.infer(tensor)
        fp16_all.extend(coco_predictions(int(image_info["id"]), decode(fp16_raw, image.shape[:2], labels, args.evaluation_confidence, args.nms_iou_threshold, args.top_k), class_to_category))
        int8_all.extend(coco_predictions(int(image_info["id"]), decode(int8_raw, image.shape[:2], labels, args.evaluation_confidence, args.nms_iou_threshold, args.top_k), class_to_category))
        fp16_operating.extend(coco_predictions(int(image_info["id"]), decode(fp16_raw, image.shape[:2], labels, args.fp16_operating_threshold, args.nms_iou_threshold, args.top_k), class_to_category))
        for threshold in int8_thresholds:
            int8_operating[threshold].extend(coco_predictions(int(image_info["id"]), decode(int8_raw, image.shape[:2], labels, threshold, args.nms_iou_threshold, args.top_k), class_to_category))
        if index % 25 == 0 or index == len(images):
            print(f"Evaluated {index}/{len(images)} COCO images")
    (output_dir / "fp16_predictions.json").write_text(json.dumps(fp16_all) + "\n", encoding="utf-8")
    (output_dir / "int8_predictions.json").write_text(json.dumps(int8_all) + "\n", encoding="utf-8")
    metrics = {
        "fp16_all_classes": coco_metrics(coco_gt, fp16_all, image_ids, category_ids),
        "int8_all_classes": coco_metrics(coco_gt, int8_all, image_ids, category_ids),
        "fp16_person": coco_metrics(coco_gt, fp16_all, image_ids, [person_category_id]),
        "int8_person": coco_metrics(coco_gt, int8_all, image_ids, [person_category_id]),
    }
    operating = {
        "fp16": fixed_threshold_person_metrics(fp16_operating, annotations, person_category_id=person_category_id, score_threshold=args.fp16_operating_threshold),
        "int8": {str(threshold): fixed_threshold_person_metrics(rows, annotations, person_category_id=person_category_id, score_threshold=threshold) for threshold, rows in int8_operating.items()},
    }
    summary = {
        "schema_version": 1,
        "purpose": "label-based primary detector quality comparison on COCO val2017",
        "dataset": {"images_dir": str(images_dir), "annotations": str(annotation_path), "images_evaluated": len(images), "full_val2017": len(images) == 5000},
        "engines": {"fp16": str(fp16_engine), "fp16_sha256": sha256(fp16_engine), "int8": str(int8_engine), "int8_sha256": sha256(int8_engine)},
        "model_labels": {"count": len(labels), "person_model_class_id": 0, "person_coco_category_id": person_category_id},
        "postprocess": {"input": "RGB letterbox 640x640, CHW float32 [0,1]", "primary_output": "[x1,y1,x2,y2,score,class_id]", "nms_iou_threshold": args.nms_iou_threshold, "top_k": args.top_k, "evaluation_confidence": args.evaluation_confidence},
        "coco_metrics": metrics,
        "person_operating_point_iou_0_50_non_crowd": operating,
        "prediction_files": {"fp16": str(output_dir / "fp16_predictions.json"), "int8": str(output_dir / "int8_predictions.json")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"coco_metrics": metrics, "person_operating_point": operating}, ensure_ascii=False, indent=2))
    print(f"Wrote COCO evaluation summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
