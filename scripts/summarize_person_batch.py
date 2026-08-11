#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a batch of person analytics outputs.")
    parser.add_argument("batch_dir", type=Path, help="Batch output directory.")
    parser.add_argument("output_json", type=Path, help="Output batch summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_batch(args.batch_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(summary, args.output_json)
    return 0


def summarize_batch(batch_dir: Path) -> dict[str, Any]:
    if not batch_dir.exists():
        raise FileNotFoundError(batch_dir)

    videos: list[dict[str, Any]] = []
    total_unique_persons_sum = 0
    line_crossing_in_sum = 0
    line_crossing_out_sum = 0
    processed_count = 0
    failed_count = 0
    batch_jobs: int | None = None
    total_frame_count = 0
    started_values: list[datetime] = []
    finished_values: list[datetime] = []

    for metadata_path in sorted(batch_dir.glob("*/run_metadata.json")):
        metadata = _read_json(metadata_path)
        item = _summarize_item(metadata_path.parent, metadata)
        videos.append(item)
        if batch_jobs is None and metadata.get("batch_jobs") is not None:
            batch_jobs = int(metadata["batch_jobs"])
        if item.get("started_at"):
            started_values.append(_parse_datetime(str(item["started_at"])))
        if item.get("finished_at"):
            finished_values.append(_parse_datetime(str(item["finished_at"])))
        if item["status"] == "ok":
            processed_count += 1
            total_frame_count += int(item.get("total_frame_count", 0))
            total_unique_persons_sum += int(item.get("total_unique_persons", 0))
            line_crossing_in_sum += int(item.get("line_crossing_in", 0))
            line_crossing_out_sum += int(item.get("line_crossing_out", 0))
        else:
            failed_count += 1

    total_duration_seconds = 0.0
    if started_values and finished_values:
        total_duration_seconds = max(0.0, (max(finished_values) - min(started_values)).total_seconds())

    return {
        "batch_dir": str(batch_dir),
        "video_count": len(videos),
        "processed_count": processed_count,
        "failed_count": failed_count,
        "batch_jobs": batch_jobs,
        "total_duration_seconds": round(total_duration_seconds, 3),
        "total_frame_count": total_frame_count,
        "processing_fps": _fps(total_frame_count, total_duration_seconds),
        "total_unique_persons_sum": total_unique_persons_sum,
        "line_crossing_in_sum": line_crossing_in_sum,
        "line_crossing_out_sum": line_crossing_out_sum,
        "videos": videos,
    }


def _summarize_item(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    summary_path = output_dir / "analytics_summary.json"
    base = {
        "input_video": metadata.get("input_video"),
        "output_dir": str(output_dir),
        "output_video": str(output_dir / "person_analytics.mp4"),
        "output_jsonl": str(output_dir / "results.jsonl"),
        "output_summary": str(summary_path),
        "output_overlay_video": str(output_dir / "person_analytics_overlay.mp4"),
        "status": metadata.get("status", "unknown"),
        "exit_code": metadata.get("exit_code"),
        "started_at": metadata.get("started_at"),
        "finished_at": metadata.get("finished_at"),
        "duration_seconds": _duration_seconds(metadata.get("started_at"), metadata.get("finished_at")),
        "log_path": metadata.get("log_path", str(output_dir / "run.log")),
    }
    base["file_sizes"] = _output_file_sizes(base)
    if base["status"] != "ok" or not summary_path.exists():
        base["error"] = metadata.get("error", "analytics_summary.json missing")
        return base

    summary = _read_json(summary_path)
    lines = summary.get("lines", [])
    rois = summary.get("rois", [])
    timeline = summary.get("timeline", {})
    stream_summaries = timeline.get("streams", {})
    total_frame_count = sum(int(stream.get("frame_count", 0)) for stream in stream_summaries.values())
    duration_seconds = base.get("duration_seconds")

    base.update(
        {
            "total_unique_persons": summary.get("global", {}).get("total_unique_persons", 0),
            "total_frame_count": total_frame_count,
            "processing_fps": _fps(total_frame_count, duration_seconds),
            "line_crossing_in": sum(int(line.get("line_crossing_in", 0)) for line in lines),
            "line_crossing_out": sum(int(line.get("line_crossing_out", 0)) for line in lines),
            "roi_unique_persons": {
                roi.get("roi_id", f"roi-{index}"): roi.get("unique_persons_in_roi", 0)
                for index, roi in enumerate(rois)
            },
            "streams": {
                stream_id: {
                    "frame_count": stream.get("frame_count", 0),
                    "is_frame_continuous": stream.get("is_frame_continuous", False),
                    "estimated_fps": stream.get("estimated_fps"),
                }
                for stream_id, stream in stream_summaries.items()
            },
        }
    )
    return base


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _duration_seconds(started_at: Any, finished_at: Any) -> float | None:
    if not started_at or not finished_at:
        return None
    return round(max(0.0, (_parse_datetime(str(finished_at)) - _parse_datetime(str(started_at))).total_seconds()), 3)


def _fps(frame_count: Any, duration_seconds: Any) -> float | None:
    frames = int(frame_count or 0)
    duration = float(duration_seconds or 0)
    if frames <= 0 or duration <= 0:
        return None
    return round(frames / duration, 3)


def _output_file_sizes(item: dict[str, Any]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for key in ("output_video", "output_overlay_video", "output_jsonl", "output_summary", "log_path"):
        raw_path = item.get(key)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if path.exists() and path.is_file():
            sizes[key] = path.stat().st_size
    return sizes


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote batch summary: {output_path}")
    print(f"Videos: {summary['video_count']}")
    print(f"Processed: {summary['processed_count']}")
    print(f"Failed: {summary['failed_count']}")
    print(f"Parallel jobs: {summary['batch_jobs']}")
    print(f"Total duration seconds: {summary['total_duration_seconds']}")
    print(f"Total frames: {summary['total_frame_count']}")
    print(f"Processing FPS: {summary['processing_fps']}")
    print(f"Unique persons sum: {summary['total_unique_persons_sum']}")
    print(f"Line in/out sum: {summary['line_crossing_in_sum']}/{summary['line_crossing_out_sum']}")


if __name__ == "__main__":
    raise SystemExit(main())
