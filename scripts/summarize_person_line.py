#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize person line crossings from tracker JSONL.")
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL path from person tracker.")
    parser.add_argument("output_json", type=Path, help="Output line crossing summary JSON path.")
    parser.add_argument(
        "--line",
        default="640,0,640,720",
        help="Line segment as x1,y1,x2,y2 in output-frame coordinates.",
    )
    parser.add_argument("--line-id", default="line-1", help="Line identifier in the output summary.")
    parser.add_argument(
        "--min-side-distance",
        type=float,
        default=1.0,
        help="Ignore points with absolute signed line distance below this value.",
    )
    parser.add_argument(
        "--count-once-per-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count only the first crossing event for each track.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    line = parse_line(args.line)
    summary = summarize_line_crossings(
        args.input_jsonl,
        line=line,
        line_id=args.line_id,
        min_side_distance=args.min_side_distance,
        count_once_per_track=args.count_once_per_track,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(summary, args.output_json)
    return 0


def parse_line(text: str) -> dict[str, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Line must be formatted as x1,y1,x2,y2")
    x1, y1, x2, y2 = (float(part) for part in parts)
    if x1 == x2 and y1 == y2:
        raise ValueError("Line endpoints must be different")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def summarize_line_crossings(
    path: Path,
    *,
    line: dict[str, float],
    line_id: str = "line-1",
    min_side_distance: float = 1.0,
    count_once_per_track: bool = True,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if min_side_distance < 0:
        raise ValueError("min_side_distance must be greater than or equal to zero")

    total_frames = 0
    track_states: dict[int, dict[str, Any]] = {}
    counted_track_ids: set[int] = set()
    crossings: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_text in f:
            if not line_text.strip():
                continue
            payload = json.loads(line_text)
            total_frames += 1
            frame_id = int(payload.get("frame_id", total_frames - 1))
            timestamp = payload.get("timestamp")

            for track in _unique_frame_tracks(payload.get("tracks", [])):
                hit = _track_line_hit(track, line, min_side_distance)
                if hit is None:
                    continue
                track_id = int(hit["track_id"])
                previous = track_states.get(track_id)
                current_side = int(hit["side"])

                if (
                    previous is not None
                    and previous["side"] != current_side
                    and (not count_once_per_track or track_id not in counted_track_ids)
                ):
                    direction = "in" if previous["side"] < current_side else "out"
                    crossings.append(
                        {
                            "track_id": track_id,
                            "direction": direction,
                            "from_side": previous["side"],
                            "to_side": current_side,
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "previous_frame_id": previous["frame_id"],
                            "previous_center": previous["center"],
                            "center": hit["center"],
                            "confidence": hit["confidence"],
                        }
                    )
                    counted_track_ids.add(track_id)

                track_states[track_id] = {
                    "side": current_side,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "center": hit["center"],
                }

    in_track_ids = sorted({item["track_id"] for item in crossings if item["direction"] == "in"})
    out_track_ids = sorted({item["track_id"] for item in crossings if item["direction"] == "out"})

    return {
        "input_jsonl": str(path),
        "line_id": line_id,
        "line": line,
        "min_side_distance": min_side_distance,
        "count_once_per_track": count_once_per_track,
        "total_frames": total_frames,
        "line_crossing_in": len(in_track_ids),
        "line_crossing_out": len(out_track_ids),
        "in_track_ids": in_track_ids,
        "out_track_ids": out_track_ids,
        "crossing_count": len(crossings),
        "crossings": crossings,
    }


def _unique_frame_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[int, dict[str, Any]] = {}
    for track in tracks:
        try:
            track_id = int(track.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if track_id < 0:
            continue
        unique[track_id] = track
    return list(unique.values())


def _track_line_hit(
    track: dict[str, Any],
    line: dict[str, float],
    min_side_distance: float,
) -> dict[str, Any] | None:
    bbox = track.get("bbox", {})
    width = float(bbox.get("width", 0.0))
    height = float(bbox.get("height", 0.0))
    if width <= 0.0 or height <= 0.0:
        return None

    center = {
        "x": float(bbox.get("left", 0.0)) + width / 2.0,
        "y": float(bbox.get("top", 0.0)) + height / 2.0,
    }
    signed_distance = _signed_line_value(center, line)
    if abs(signed_distance) < min_side_distance:
        return None

    return {
        "track_id": int(track["track_id"]),
        "class_id": int(track.get("class_id", 0)),
        "confidence": float(track.get("confidence", 0.0)),
        "center": center,
        "side": 1 if signed_distance > 0 else -1,
        "signed_distance": signed_distance,
    }


def _signed_line_value(point: dict[str, float], line: dict[str, float]) -> float:
    dx = line["x2"] - line["x1"]
    dy = line["y2"] - line["y1"]
    return dx * (point["y"] - line["y1"]) - dy * (point["x"] - line["x1"])


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote line summary: {output_path}")
    print(f"Line: {summary['line_id']} {summary['line']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Line crossing in: {summary['line_crossing_in']}")
    print(f"Line crossing out: {summary['line_crossing_out']}")
    print(f"Crossing events: {summary['crossing_count']}")


if __name__ == "__main__":
    raise SystemExit(main())
