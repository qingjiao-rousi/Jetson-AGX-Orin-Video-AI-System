#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize person track JSONL output.")
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL path from person tracker.")
    parser.add_argument("output_json", type=Path, help="Output summary JSON path.")
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=2,
        help="Minimum frames required for a track to count as a unique person.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_tracks(args.input_jsonl, min_track_frames=args.min_track_frames)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(summary, args.output_json)
    return 0


def summarize_tracks(path: Path, *, min_track_frames: int = 2) -> dict[str, Any]:
    if min_track_frames <= 0:
        raise ValueError("min_track_frames must be greater than zero")
    if not path.exists():
        raise FileNotFoundError(path)

    total_lines = 0
    total_frames = 0
    frames_with_person = 0
    frames_with_tracks = 0
    max_persons_in_frame = 0
    max_tracks_in_frame = 0
    detection_count = 0
    track_observation_count = 0
    tracks: dict[int, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_lines += 1
            payload = json.loads(line)
            total_frames += 1
            frame_id = int(payload.get("frame_id", total_frames - 1))
            timestamp = payload.get("timestamp")
            detections = payload.get("detections", [])
            frame_tracks = _unique_frame_tracks(payload.get("tracks", []))

            detection_count += len(detections)
            track_observation_count += len(frame_tracks)
            max_persons_in_frame = max(max_persons_in_frame, len(detections))
            max_tracks_in_frame = max(max_tracks_in_frame, len(frame_tracks))
            if detections:
                frames_with_person += 1
            if frame_tracks:
                frames_with_tracks += 1

            for track in frame_tracks:
                track_id = int(track["track_id"])
                bbox = track.get("bbox", {})
                entry = tracks.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "first_frame": frame_id,
                        "last_frame": frame_id,
                        "first_timestamp": timestamp,
                        "last_timestamp": timestamp,
                        "frame_count": 0,
                        "class_id": int(track.get("class_id", 0)),
                        "last_bbox": bbox,
                    },
                )
                entry["first_frame"] = min(entry["first_frame"], frame_id)
                entry["last_frame"] = max(entry["last_frame"], frame_id)
                entry["last_timestamp"] = timestamp
                entry["frame_count"] += 1
                entry["last_bbox"] = bbox

    stable_tracks = [
        track for track in tracks.values() if int(track["frame_count"]) >= min_track_frames
    ]
    stable_tracks.sort(key=lambda item: (item["first_frame"], item["track_id"]))

    return {
        "input_jsonl": str(path),
        "min_track_frames": min_track_frames,
        "total_lines": total_lines,
        "total_frames": total_frames,
        "frames_with_person": frames_with_person,
        "frames_with_tracks": frames_with_tracks,
        "total_detections": detection_count,
        "total_track_observations": track_observation_count,
        "max_persons_in_frame": max_persons_in_frame,
        "max_tracks_in_frame": max_tracks_in_frame,
        "total_unique_persons": len(stable_tracks),
        "all_track_ids": sorted(tracks),
        "stable_track_ids": [track["track_id"] for track in stable_tracks],
        "tracks": stable_tracks,
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


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote summary: {output_path}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Frames with person: {summary['frames_with_person']}")
    print(f"Unique persons: {summary['total_unique_persons']}")
    print(f"Max persons in frame: {summary['max_persons_in_frame']}")
    print(f"Stable track IDs: {summary['stable_track_ids'][:20]}")


if __name__ == "__main__":
    raise SystemExit(main())
