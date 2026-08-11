#!/usr/bin/env python3
"""Compare primary-YOLO FP16 and INT8 detector outputs on fixed local frames.

This is deliberately an *offline detector* comparison.  It does not start a
DeepStream pipeline and does not involve tracking, routing, specialist models,
queues, sinks, or frame dropping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.application.detection_alignment import match_detections
from app.application.helmet_service import TensorRTHelmetBackend, letterbox, load_labels
from app.domain.entities import Detection
from app.application.primary_yolo_decoder import decode_primary_yolov8_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Image/video files or directories, in deterministic order.")
    parser.add_argument("--fp16-engine", type=Path, default=Path("models/fp16/yolov8s.engine"))
    parser.add_argument("--int8-engine", type=Path, default=Path("models/int8/yolov8s_coco_train504.engine"))
    parser.add_argument("--labels", type=Path, default=Path("models/labels.txt"))
    parser.add_argument("--class-ids", default="0", help="Comma-separated retained classes; default 0 matches the person-only deployment.")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--video-stride", type=int, default=30, help="Keep every Nth decoded video frame.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum total frames; 0 means unlimited.")
    parser.add_argument("--output", type=Path, help="Output directory (defaults below outputs/precision_alignment/).")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_class_ids(value: str) -> set[int]:
    try:
        values = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("--class-ids must be comma-separated integers") from exc
    if any(item < 0 for item in values):
        raise ValueError("--class-ids must not contain negative values")
    return values


def image_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in inputs:
        path = resolve(raw_path)
        if path.is_dir():
            paths.extend(sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES))
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    return paths


def iter_frames(inputs: list[Path], video_stride: int) -> Iterator[tuple[str, np.ndarray]]:
    if video_stride <= 0:
        raise ValueError("--video-stride must be greater than zero")
    for path in image_paths(inputs):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read image: {path}")
        yield str(path), image
    for raw_path in inputs:
        path = resolve(raw_path)
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % video_stride == 0:
                    yield f"{path}#frame={frame_index}", frame
                frame_index += 1
        finally:
            capture.release()


def prepare(image: np.ndarray) -> np.ndarray:
    model_image, _, _, _ = letterbox(image, 640, 640)
    rgb = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])


def decode(
    output: np.ndarray,
    source_shape: tuple[int, int],
    labels: tuple[str, ...],
    confidence_threshold: float,
    nms_iou_threshold: float,
    class_ids: set[int],
) -> list[Detection]:
    raw = decode_primary_yolov8_output(
        output,
        image_shape=source_shape,
        labels=labels,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
    return [
        Detection(item.class_id, item.class_name, item.confidence, item.bbox)
        for item in raw
        if item.class_id in class_ids
    ]


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), value)), 6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialise_detection(detection: Detection) -> dict[str, object]:
    return {
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": round(detection.confidence, 6),
        "bbox": {
            "left": round(detection.bbox.left, 3), "top": round(detection.bbox.top, 3),
            "width": round(detection.bbox.width, 3), "height": round(detection.bbox.height, 3),
        },
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0 or not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise SystemExit("confidence/NMS IoU thresholds must be between 0 and 1")
    if not 0.0 <= args.match_iou_threshold <= 1.0:
        raise SystemExit("--match-iou-threshold must be between 0 and 1")
    class_ids = parse_class_ids(args.class_ids)
    fp16_engine, int8_engine, labels_path = (resolve(args.fp16_engine), resolve(args.int8_engine), resolve(args.labels))
    for path, label in ((fp16_engine, "FP16 engine"), (int8_engine, "INT8 engine"), (labels_path, "labels")):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve(args.output) if args.output else PROJECT_ROOT / "outputs" / "precision_alignment" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(str(labels_path))
    fp16_backend = TensorRTHelmetBackend(str(fp16_engine))
    int8_backend = TensorRTHelmetBackend(str(int8_engine))
    total_fp16 = total_int8 = total_matched = 0
    ious: list[float] = []
    confidence_deltas: list[float] = []
    frames = 0
    jsonl_path = output_dir / "frame_comparisons.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for source, image in iter_frames(args.input, args.video_stride):
            if args.max_frames and frames >= args.max_frames:
                break
            tensor = prepare(image)
            fp16 = decode(fp16_backend.infer(tensor), image.shape[:2], labels, args.confidence_threshold, args.nms_iou_threshold, class_ids)
            int8 = decode(int8_backend.infer(tensor), image.shape[:2], labels, args.confidence_threshold, args.nms_iou_threshold, class_ids)
            matches, fp16_only, int8_only = match_detections(fp16, int8, iou_threshold=args.match_iou_threshold)
            total_fp16 += len(fp16)
            total_int8 += len(int8)
            total_matched += len(matches)
            ious.extend(item.iou for item in matches)
            confidence_deltas.extend(item.confidence_delta for item in matches)
            handle.write(json.dumps({
                "source": source,
                "fp16": [serialise_detection(item) for item in fp16],
                "int8": [serialise_detection(item) for item in int8],
                "matches": [{"fp16_index": item.reference_index, "int8_index": item.candidate_index, "iou": round(item.iou, 6), "confidence_delta": round(item.confidence_delta, 6)} for item in matches],
                "fp16_only_indices": list(fp16_only), "int8_only_indices": list(int8_only),
            }, ensure_ascii=False) + "\n")
            frames += 1
            print(f"Compared {frames}: {source} | fp16={len(fp16)} int8={len(int8)} matched={len(matches)}")
    if frames == 0:
        raise SystemExit("no readable frames found; provide image files/directories or video files")
    summary = {
        "schema_version": 1,
        "purpose": "offline primary-detector FP16/INT8 alignment before tracking/routing/specialist workers",
        "frames": frames,
        "input": [str(resolve(path)) for path in args.input],
        "engines": {"fp16": str(fp16_engine), "fp16_sha256": sha256(fp16_engine), "int8": str(int8_engine), "int8_sha256": sha256(int8_engine)},
        "postprocess": {"input": "RGB letterbox 640x640, CHW float32 [0,1]", "primary_output": "[x1,y1,x2,y2,score,class_id]", "confidence_threshold": args.confidence_threshold, "nms_iou_threshold": args.nms_iou_threshold, "retained_class_ids": sorted(class_ids), "match_iou_threshold": args.match_iou_threshold},
        "detections": {"fp16_total": total_fp16, "int8_total": total_int8, "matched": total_matched, "fp16_only": total_fp16 - total_matched, "int8_only": total_int8 - total_matched, "fp16_match_rate": round(total_matched / total_fp16, 6) if total_fp16 else None, "int8_match_rate": round(total_matched / total_int8, 6) if total_int8 else None, "symmetric_match_rate": round(2 * total_matched / (total_fp16 + total_int8), 6) if total_fp16 + total_int8 else None},
        "matched_box_iou": {"samples": len(ious), "p50": percentile(ious, 50), "p95": percentile(ious, 95), "mean": round(float(np.mean(ious)), 6) if ious else None},
        "confidence_delta_int8_minus_fp16": {"samples": len(confidence_deltas), "p50": percentile(confidence_deltas, 50), "p95": percentile(confidence_deltas, 95), "mean": round(float(np.mean(confidence_deltas)), 6) if confidence_deltas else None, "mean_absolute": round(float(np.mean(np.abs(confidence_deltas))), 6) if confidence_deltas else None},
        "frame_comparisons": str(jsonl_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["detections"], ensure_ascii=False))
    print(f"Wrote primary alignment summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
