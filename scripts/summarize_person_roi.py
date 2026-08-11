#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize unique person tracks inside an ROI.")
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL path from person tracker.")
    parser.add_argument("output_json", type=Path, help="Output ROI summary JSON path.")
    parser.add_argument(
        "--roi",
        default="0,0,1280,720",
        help="ROI rectangle as x,y,width,height in output-frame coordinates.",
    )
    parser.add_argument("--roi-id", default="roi-1", help="ROI identifier in the output summary.")
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=2,
        help="Minimum frames inside the ROI required to count a person.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roi = parse_roi(args.roi)
    summary = summarize_roi(
        args.input_jsonl,
        roi=roi,
        roi_id=args.roi_id,
        min_track_frames=args.min_track_frames,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(summary, args.output_json)
    return 0


def parse_roi(text: str) -> dict[str, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be formatted as x,y,width,height")
    x, y, width, height = (float(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be greater than zero")
    return {"x": x, "y": y, "width": width, "height": height}


def summarize_roi(
    path: Path,
    *,
    roi: dict[str, float],
    roi_id: str = "roi-1",
    min_track_frames: int = 2,
) -> dict[str, Any]:
    if min_track_frames <= 0:
        raise ValueError("min_track_frames must be greater than zero")
    if not path.exists():
        raise FileNotFoundError(path)

    total_frames = 0
    frames_with_roi_person = 0
    max_persons_in_roi_frame = 0
    roi_observations = 0
    tracks: dict[int, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            total_frames += 1
            frame_id = int(payload.get("frame_id", total_frames - 1))
            timestamp = payload.get("timestamp")
            frame_hits: dict[int, dict[str, Any]] = {}

            for track in payload.get("tracks", []):
                hit = _track_roi_hit(track, roi)
                if hit is None:
                    continue
                track_id = int(track["track_id"])
                frame_hits[track_id] = hit

            if frame_hits:
                frames_with_roi_person += 1
                max_persons_in_roi_frame = max(max_persons_in_roi_frame, len(frame_hits))
            roi_observations += len(frame_hits)

            for track_id, hit in frame_hits.items():
                entry = tracks.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "first_frame": frame_id,
                        "last_frame": frame_id,
                        "first_timestamp": timestamp,
                        "last_timestamp": timestamp,
                        "frames_in_roi": 0,
                        "class_id": int(hit.get("class_id", 0)),
                        "confidence_sum": 0.0,
                        "max_confidence": 0.0,
                        "last_center": hit["center"],
                        "last_bbox": hit["bbox"],
                    },
                )
                entry["first_frame"] = min(entry["first_frame"], frame_id)
                entry["last_frame"] = max(entry["last_frame"], frame_id)
                entry["last_timestamp"] = timestamp
                entry["frames_in_roi"] += 1
                entry["confidence_sum"] += float(hit.get("confidence", 0.0))
                entry["max_confidence"] = max(
                    float(entry["max_confidence"]), float(hit.get("confidence", 0.0))
                )
                entry["last_center"] = hit["center"]
                entry["last_bbox"] = hit["bbox"]

    stable_tracks = [
        _finalize_track(track)
        for track in tracks.values()
        if int(track["frames_in_roi"]) >= min_track_frames
    ]
    stable_tracks.sort(key=lambda item: (item["first_frame"], item["track_id"]))

    return {
        "input_jsonl": str(path),
        "roi_id": roi_id,
        "roi": roi,
        "min_track_frames": min_track_frames,
        "total_frames": total_frames,
        "frames_with_roi_person": frames_with_roi_person,
        "roi_observations": roi_observations,
        "max_persons_in_roi_frame": max_persons_in_roi_frame,
        "unique_persons_in_roi": len(stable_tracks),
        "stable_track_ids": [track["track_id"] for track in stable_tracks],
        "tracks": stable_tracks,
    }


def _track_roi_hit(track: dict[str, Any], roi: dict[str, float]) -> dict[str, Any] | None:
    try:
        track_id = int(track.get("track_id", -1))
    except (TypeError, ValueError):
        return None
    if track_id < 0:
        return None

    bbox = track.get("bbox", {})
    left = float(bbox.get("left", 0.0))
    top = float(bbox.get("top", 0.0))
    width = float(bbox.get("width", 0.0))
    height = float(bbox.get("height", 0.0))
    if width <= 0.0 or height <= 0.0:
        return None

    center = {"x": left + width / 2.0, "y": top + height / 2.0}
    if not _point_in_roi(center, roi):
        return None

    return {
        "track_id": track_id,
        "class_id": int(track.get("class_id", 0)),
        "confidence": float(track.get("confidence", 0.0)),
        "bbox": bbox,
        "center": center,
    }


def _point_in_roi(point: dict[str, float], roi: dict[str, float]) -> bool:
    return (
        roi["x"] <= point["x"] <= roi["x"] + roi["width"]
        and roi["y"] <= point["y"] <= roi["y"] + roi["height"]
    )


def _finalize_track(track: dict[str, Any]) -> dict[str, Any]:
    result = dict(track)
    frames = max(int(result.pop("frames_in_roi")), 1)
    confidence_sum = float(result.pop("confidence_sum"))
    result["frames_in_roi"] = frames
    result["average_confidence"] = confidence_sum / frames
    return result


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote ROI summary: {output_path}")
    print(f"ROI: {summary['roi_id']} {summary['roi']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Frames with ROI person: {summary['frames_with_roi_person']}")
    print(f"Unique persons in ROI: {summary['unique_persons_in_roi']}")
    print(f"Max persons in ROI frame: {summary['max_persons_in_roi_frame']}")
    print(f"Stable track IDs: {summary['stable_track_ids'][:20]}")


if __name__ == "__main__":
    raise SystemExit(main())
