#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize frame timeline and continuity from JSONL.")
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL path.")
    parser.add_argument("output_json", type=Path, help="Output timeline summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_timeline(args.input_jsonl)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(summary, args.output_json)
    return 0


def summarize_timeline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    streams: dict[str, dict[str, Any]] = {}
    total_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_lines += 1
            payload = json.loads(line)
            stream_id = str(payload.get("stream_id", "stream-0"))
            frame_id = int(payload.get("frame_id", total_lines - 1))
            timestamp_text = payload.get("timestamp")
            timestamp = parse_timestamp(timestamp_text)

            stream = streams.setdefault(
                stream_id,
                {
                    "stream_id": stream_id,
                    "frame_ids": [],
                    "timestamps": [],
                    "timestamp_texts": [],
                    "non_monotonic_timestamp_frames": [],
                },
            )
            stream["frame_ids"].append(frame_id)
            stream["timestamp_texts"].append(timestamp_text)
            stream["timestamps"].append(timestamp)

    stream_summaries = {
        stream_id: _summarize_stream(stream)
        for stream_id, stream in sorted(streams.items())
    }

    return {
        "input_jsonl": str(path),
        "total_lines": total_lines,
        "stream_count": len(stream_summaries),
        "streams": stream_summaries,
    }


def parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _summarize_stream(stream: dict[str, Any]) -> dict[str, Any]:
    frame_ids = list(stream["frame_ids"])
    timestamps = list(stream["timestamps"])
    timestamp_texts = list(stream["timestamp_texts"])

    frame_count = len(frame_ids)
    first_frame = min(frame_ids) if frame_ids else None
    last_frame = max(frame_ids) if frame_ids else None
    expected_frames = (
        list(range(first_frame, last_frame + 1))
        if first_frame is not None and last_frame is not None
        else []
    )
    frame_id_set = set(frame_ids)
    missing_frames = [frame_id for frame_id in expected_frames if frame_id not in frame_id_set]
    duplicate_frames = sorted(
        frame_id for frame_id in frame_id_set if frame_ids.count(frame_id) > 1
    )
    out_of_order_frames = [
        {"index": index, "previous": frame_ids[index - 1], "current": frame_ids[index]}
        for index in range(1, len(frame_ids))
        if frame_ids[index] < frame_ids[index - 1]
    ]

    parsed_pairs = [
        (frame_id, timestamp)
        for frame_id, timestamp in zip(frame_ids, timestamps)
        if timestamp is not None
    ]
    non_monotonic_timestamp_frames = []
    for index in range(1, len(parsed_pairs)):
        previous_frame, previous_ts = parsed_pairs[index - 1]
        current_frame, current_ts = parsed_pairs[index]
        if current_ts < previous_ts:
            non_monotonic_timestamp_frames.append(
                {
                    "previous_frame": previous_frame,
                    "current_frame": current_frame,
                    "previous_timestamp": previous_ts.isoformat(),
                    "current_timestamp": current_ts.isoformat(),
                }
            )

    first_timestamp = parsed_pairs[0][1] if parsed_pairs else None
    last_timestamp = parsed_pairs[-1][1] if parsed_pairs else None
    duration_seconds = (
        (last_timestamp - first_timestamp).total_seconds()
        if first_timestamp is not None and last_timestamp is not None
        else None
    )
    estimated_fps = (
        (frame_count - 1) / duration_seconds
        if duration_seconds and duration_seconds > 0 and frame_count > 1
        else None
    )

    return {
        "stream_id": stream["stream_id"],
        "frame_count": frame_count,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "expected_frame_count": len(expected_frames),
        "missing_frame_count": len(missing_frames),
        "missing_frames": missing_frames[:100],
        "duplicate_frame_count": len(duplicate_frames),
        "duplicate_frames": duplicate_frames[:100],
        "out_of_order_frame_count": len(out_of_order_frames),
        "out_of_order_frames": out_of_order_frames[:100],
        "first_timestamp": first_timestamp.isoformat() if first_timestamp else None,
        "last_timestamp": last_timestamp.isoformat() if last_timestamp else None,
        "duration_seconds": duration_seconds,
        "estimated_fps": estimated_fps,
        "timestamp_parse_count": len(parsed_pairs),
        "timestamp_missing_count": len([item for item in timestamp_texts if not item]),
        "non_monotonic_timestamp_count": len(non_monotonic_timestamp_frames),
        "non_monotonic_timestamp_frames": non_monotonic_timestamp_frames[:100],
        "is_frame_continuous": not missing_frames and not duplicate_frames and not out_of_order_frames,
        "is_timestamp_monotonic": not non_monotonic_timestamp_frames,
    }


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote timeline summary: {output_path}")
    for stream_id, stream in summary["streams"].items():
        print(
            f"{stream_id}: frames={stream['frame_count']} "
            f"continuous={stream['is_frame_continuous']} "
            f"fps={stream['estimated_fps']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
