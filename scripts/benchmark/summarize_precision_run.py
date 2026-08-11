#!/usr/bin/env python3
"""Summarize FP16/INT8/mixed-precision runtime and prediction outputs.

Example:
  python3 scripts/benchmark/summarize_precision_run.py \
    --run fp16=outputs/fp16_realtime_8streams_4k \
    --run int8=outputs/int8_realtime_8streams_4k \
    --run mixed=outputs/mixed_realtime_8streams_4k \
    --output outputs/precision_summary.json

This script reports runtime and prediction counts. mAP, precision, recall,
false positives and false negatives require ground-truth annotations and are
therefore marked as unavailable here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def average(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(mean(values), 4) if values else None


def nested_value(payload: dict[str, Any], *keys: str, default=None):
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def event_signature(event: dict[str, Any]) -> tuple[object, ...]:
    """Stable event identity for cross-run consistency checks."""
    return (
        event.get("event_type"),
        event.get("stream_id"),
        event.get("track_id"),
        event.get("frame_id"),
        event.get("status"),
    )


def summarize(name: str, output_dir: Path, warmup_samples: int) -> dict[str, Any]:
    metrics = read_jsonl(output_dir / "runtime_metrics.jsonl")
    results = read_jsonl(output_dir / "results.jsonl")
    events = read_jsonl(output_dir / "events.jsonl")

    usable_metrics = metrics[warmup_samples:] if len(metrics) > warmup_samples else metrics
    last = metrics[-1] if metrics else {}
    latency = nested_value(last, "latency", default={})
    queues = nested_value(last, "queues", default={})
    controls = nested_value(last, "controls", default={})

    detection_totals: Counter[str] = Counter()
    track_totals: Counter[str] = Counter()
    result_frames: Counter[str] = Counter()
    detection_frames: Counter[str] = Counter()
    for row in results:
        stream_id = str(row.get("stream_id", "unknown"))
        detections = row.get("detections", [])
        tracks = row.get("tracks", [])
        result_frames[stream_id] += 1
        detection_totals[stream_id] += len(detections) if isinstance(detections, list) else 0
        track_totals[stream_id] += len(tracks) if isinstance(tracks, list) else 0
        if detections:
            detection_frames[stream_id] += 1

    event_counts = Counter(str(row.get("event_type", "unknown")) for row in events)
    event_signatures = sorted(event_signature(row) for row in events)
    stream_metrics = last.get("streams", {}) if isinstance(last, dict) else {}
    streams: dict[str, Any] = {}
    stream_ids = set(stream_metrics) | set(result_frames)
    for stream_id in sorted(stream_ids):
        item = stream_metrics.get(stream_id, {})
        streams[stream_id] = {
            "result_frames": result_frames.get(stream_id, 0),
            "frames_with_detections": detection_frames.get(stream_id, 0),
            "total_detections": detection_totals.get(stream_id, 0),
            "total_tracks": track_totals.get(stream_id, 0),
            "estimated_processing_fps": item.get("estimated_processing_fps"),
            "dropped_frames_estimated": item.get("dropped_frames"),
            "dropped_frame_rate_estimated": item.get("dropped_frame_rate"),
            "status_at_end": item.get("status"),
            "stale_count": item.get("stale_count"),
        }

    return {
        "name": name,
        "output_dir": str(output_dir),
        "metrics_samples": len(metrics),
        "warmup_samples_ignored": min(warmup_samples, len(metrics)),
        "runtime": {
            "elapsed_seconds": last.get("elapsed_seconds"),
            "total_result_frames": last.get("total_frames"),
            "last_processing_fps": last.get("processing_fps"),
            "average_processing_fps": average(usable_metrics, "processing_fps"),
            "average_gpu_utilization_percent": average(usable_metrics, "gpu", "utilization_gpu"),
            "average_gpu_memory_utilization_percent": average(usable_metrics, "gpu", "utilization_memory"),
            "average_temperature_c": average(usable_metrics, "gpu", "temperature_c"),
            "average_gpu_soc_power_mw": average(usable_metrics, "gpu", "power_gpu_soc_mw"),
            "average_gpu_soc_power_avg_mw": average(usable_metrics, "gpu", "power_gpu_soc_avg_mw"),
            "average_ram_used_mb": average(usable_metrics, "gpu", "ram_used_mb"),
            "max_process_rss_mb": round((last.get("process", {}).get("max_rss_kb", 0) or 0) / 1024, 2),
        },
        "latency": {
            "definition": nested_value(latency, "definition"),
            "pipeline_p50_ms": nested_value(latency, "pipeline", "p50_ms"),
            "pipeline_p95_ms": nested_value(latency, "pipeline", "p95_ms"),
            "pipeline_samples": nested_value(latency, "pipeline", "samples", default=0),
            "json_writer_p50_ms": nested_value(latency, "json_writer", "p50_ms"),
            "json_writer_p95_ms": nested_value(latency, "json_writer", "p95_ms"),
            "end_to_end_p50_ms": nested_value(latency, "end_to_end", "p50_ms"),
            "end_to_end_p95_ms": nested_value(latency, "end_to_end", "p95_ms"),
            "end_to_end_samples": nested_value(latency, "end_to_end", "samples", default=0),
            "unmatched_results": nested_value(latency, "unmatched_results", default=0),
            "unmatched_writes": nested_value(latency, "unmatched_writes", default=0),
        },
        "drop_and_queue_stats": {
            "writer_dropped": nested_value(queues, "writer", "dropped", default=0),
            "writer_write_errors": nested_value(queues, "writer", "write_errors", default=0),
            "task_buffer_dropped": nested_value(queues, "task_buffer", "dropped", default=0),
            "frame_store_dropped": nested_value(queues, "frame_store", "dropped", default=0),
            "fps_controller_dropped": nested_value(controls, "fps", "dropped_frames", default=0),
            "fps_controller_drop_ratio": nested_value(controls, "fps", "drop_ratio", default=0.0),
            "backpressure_max_pending": nested_value(controls, "backpressure", "max_pending_ever", default=0),
            "task_buffer_by_task": nested_value(queues, "task_buffer", "by_task", default={}),
            "helmet_worker": nested_value(queues, "workers", "helmet", default={}),
        },
        "predictions": {
            "result_rows": len(results),
            "total_detections": sum(detection_totals.values()),
            "total_tracks": sum(track_totals.values()),
            "event_counts": dict(sorted(event_counts.items())),
            "event_signatures": [list(signature) for signature in event_signatures],
        },
        "streams": streams,
        "ground_truth_metrics": {
            "map": "unavailable_without_ground_truth",
            "precision": "unavailable_without_ground_truth",
            "recall": "unavailable_without_ground_truth",
            "false_positives": "unavailable_without_ground_truth",
            "false_negatives": "unavailable_without_ground_truth",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=OUTPUT_DIR",
        help="Run label and output directory; may be specified multiple times.",
    )
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    summary: dict[str, Any] = {"runs": []}
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run must use NAME=OUTPUT_DIR: {spec}")
        name, raw_dir = spec.split("=", 1)
        summary["runs"].append(summarize(name, Path(raw_dir), max(args.warmup_samples, 0)))

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"写入: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
