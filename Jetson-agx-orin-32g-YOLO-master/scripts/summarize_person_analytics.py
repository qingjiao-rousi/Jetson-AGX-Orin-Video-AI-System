#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from summarize_person_line import parse_line, summarize_line_crossings
from summarize_person_roi import parse_roi, summarize_roi
from summarize_person_timeline import summarize_timeline
from summarize_person_tracks import summarize_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified person analytics summary.")
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL path from person tracker.")
    parser.add_argument("config_yaml", type=Path, help="Analytics config YAML path.")
    parser.add_argument("output_json", type=Path, help="Unified analytics summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_analytics(args.input_jsonl, args.config_yaml)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(summary, args.output_json)
    return 0


def summarize_analytics(input_jsonl: Path, config_yaml: Path) -> dict[str, Any]:
    config = _load_config(config_yaml)
    min_track_frames = int(config.get("min_track_frames", 2))

    summary: dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "config_yaml": str(config_yaml),
        "min_track_frames": min_track_frames,
        "timeline": summarize_timeline(input_jsonl),
        "global": summarize_tracks(input_jsonl, min_track_frames=min_track_frames),
        "rois": [],
        "lines": [],
    }

    for roi_cfg in config.get("rois", []):
        roi = _roi_from_config(roi_cfg)
        summary["rois"].append(
            summarize_roi(
                input_jsonl,
                roi=roi,
                roi_id=str(roi_cfg["id"]),
                min_track_frames=int(roi_cfg.get("min_track_frames", min_track_frames)),
            )
        )

    for line_cfg in config.get("lines", []):
        line = _line_from_config(line_cfg)
        summary["lines"].append(
            summarize_line_crossings(
                input_jsonl,
                line=line,
                line_id=str(line_cfg["id"]),
                min_side_distance=float(line_cfg.get("min_side_distance", 1.0)),
                count_once_per_track=bool(line_cfg.get("count_once_per_track", True)),
            )
        )

    return summary


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Analytics config must be a YAML mapping")
    return data


def _roi_from_config(config: dict[str, Any]) -> dict[str, float]:
    if "id" not in config:
        raise ValueError("ROI config requires id")
    rect = config.get("rect")
    if not isinstance(rect, list) or len(rect) != 4:
        raise ValueError(f"ROI `{config['id']}` requires rect: [x, y, width, height]")
    return parse_roi(",".join(str(value) for value in rect))


def _line_from_config(config: dict[str, Any]) -> dict[str, float]:
    if "id" not in config:
        raise ValueError("Line config requires id")
    points = config.get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError(f"Line `{config['id']}` requires points: [x1, y1, x2, y2]")
    return parse_line(",".join(str(value) for value in points))


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote analytics summary: {output_path}")
    print(f"Unique persons: {summary['global']['total_unique_persons']}")
    for stream_id, stream in summary["timeline"]["streams"].items():
        print(
            f"Timeline {stream_id}: frames={stream['frame_count']} "
            f"continuous={stream['is_frame_continuous']} fps={stream['estimated_fps']}"
        )
    for roi in summary["rois"]:
        print(f"ROI {roi['roi_id']}: {roi['unique_persons_in_roi']} unique persons")
    for line in summary["lines"]:
        print(
            f"Line {line['line_id']}: in={line['line_crossing_in']} "
            f"out={line['line_crossing_out']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
