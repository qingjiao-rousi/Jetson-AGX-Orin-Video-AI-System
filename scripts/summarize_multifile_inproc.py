#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_person_timeline import summarize_timeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one in-process multi-file pipeline output directory.")
    parser.add_argument("output_dir", type=Path, help="Output directory from run_multifile_inproc.sh.")
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        help="Output summary JSON path. Defaults to multifile_summary.json in output_dir.",
    )
    parser.add_argument("--expected-stream-count", type=int, default=8)
    parser.add_argument("--min-track-frames", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = args.output_json or (args.output_dir / "multifile_summary.json")
    summary = summarize_multifile(
        args.output_dir,
        expected_stream_count=args.expected_stream_count,
        min_track_frames=args.min_track_frames,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(summary, output_json)
    return 0


def summarize_multifile(
    output_dir: Path,
    *,
    expected_stream_count: int = 8,
    min_track_frames: int = 2,
) -> dict[str, Any]:
    if expected_stream_count <= 0:
        raise ValueError("expected_stream_count must be greater than zero")
    if min_track_frames <= 0:
        raise ValueError("min_track_frames must be greater than zero")

    results_jsonl = output_dir / "results.jsonl"
    tiled_video = output_dir / "multifile_preview.mp4"
    run_log = output_dir / "run.log"
    run_metadata = output_dir / "run_metadata.json"
    runtime_config = output_dir / ".runtime" / "app_multifile_runtime.yaml"

    metadata = _read_json_or_empty(run_metadata)
    timeline = summarize_timeline(results_jsonl) if results_jsonl.exists() else {
        "input_jsonl": str(results_jsonl),
        "total_lines": 0,
        "valid_json_lines": 0,
        "malformed_json_line_count": 0,
        "malformed_json_lines": [],
        "stream_count": 0,
        "streams": {},
    }
    stream_stats, stats_malformed_json_line_count = _collect_stream_stats(
        results_jsonl,
        min_track_frames=min_track_frames,
    )
    streams = {}
    for stream_id in sorted(set(timeline.get("streams", {})) | set(stream_stats)):
        timeline_stream = dict(timeline.get("streams", {}).get(stream_id, {}))
        stats_stream = stream_stats.get(stream_id, {})
        streams[stream_id] = {
            **timeline_stream,
            **stats_stream,
        }

    total_frame_count = sum(int(stream.get("frame_count", 0)) for stream in streams.values())
    total_detections = sum(int(stream.get("total_detections", 0)) for stream in streams.values())
    total_track_observations = sum(int(stream.get("total_track_observations", 0)) for stream in streams.values())
    total_unique_persons = sum(int(stream.get("total_unique_persons", 0)) for stream in streams.values())

    return {
        "mode": "inprocess_multifile",
        "output_dir": str(output_dir),
        "expected_stream_count": expected_stream_count,
        "observed_stream_count": len(streams),
        "missing_stream_ids": _missing_stream_ids(streams, expected_stream_count),
        "results_jsonl": str(results_jsonl),
        "tiled_video": str(tiled_video),
        "run_log": str(run_log),
        "run_metadata": str(run_metadata),
        "runtime_config": str(runtime_config),
        "run_status": metadata.get("status", "unknown") if metadata else "unknown",
        "exit_code": metadata.get("exit_code") if metadata else None,
        "error": metadata.get("error", "") if metadata else "run_metadata.json missing",
        "started_at": metadata.get("started_at") if metadata else None,
        "finished_at": metadata.get("finished_at") if metadata else None,
        "input_videos": metadata.get("input_videos", []) if metadata else [],
        "file_sizes": _file_sizes(
            {
                "results_jsonl": results_jsonl,
                "tiled_video": tiled_video,
                "run_log": run_log,
                "run_metadata": run_metadata,
                "runtime_config": runtime_config,
            }
        ),
        "total_lines": int(timeline.get("total_lines", 0)),
        "valid_json_lines": int(timeline.get("valid_json_lines", timeline.get("total_lines", 0))),
        "malformed_json_line_count": max(
            int(timeline.get("malformed_json_line_count", 0)),
            stats_malformed_json_line_count,
        ),
        "malformed_json_lines": list(timeline.get("malformed_json_lines", [])),
        "total_frame_count": total_frame_count,
        "total_detections": total_detections,
        "total_track_observations": total_track_observations,
        "total_unique_persons": total_unique_persons,
        "min_track_frames": min_track_frames,
        "streams": streams,
    }


def _collect_stream_stats(path: Path, *, min_track_frames: int) -> tuple[dict[str, dict[str, Any]], int]:
    if not path.exists():
        return {}, 0

    streams: dict[str, dict[str, Any]] = {}
    malformed_count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            stream_id = str(payload.get("stream_id", "stream-0"))
            detections = payload.get("detections", [])
            tracks = payload.get("tracks", [])
            stream = streams.setdefault(
                stream_id,
                {
                    "stream_id": stream_id,
                    "frames_with_detections": 0,
                    "frames_with_tracks": 0,
                    "total_detections": 0,
                    "total_track_observations": 0,
                    "max_detections_in_frame": 0,
                    "max_tracks_in_frame": 0,
                    "_track_frames": {},
                },
            )
            detection_count = len(detections) if isinstance(detections, list) else 0
            frame_tracks = _unique_track_ids(tracks if isinstance(tracks, list) else [])
            track_count = len(frame_tracks)
            stream["total_detections"] += detection_count
            stream["total_track_observations"] += track_count
            stream["max_detections_in_frame"] = max(stream["max_detections_in_frame"], detection_count)
            stream["max_tracks_in_frame"] = max(stream["max_tracks_in_frame"], track_count)
            if detection_count:
                stream["frames_with_detections"] += 1
            if track_count:
                stream["frames_with_tracks"] += 1
            for track_id in frame_tracks:
                track_frames = stream["_track_frames"]
                track_frames[track_id] = int(track_frames.get(track_id, 0)) + 1

    for stream in streams.values():
        track_frames = dict(stream.pop("_track_frames"))
        stable_track_ids = sorted(
            track_id for track_id, frame_count in track_frames.items() if frame_count >= min_track_frames
        )
        stream["all_track_ids"] = sorted(track_frames)
        stream["stable_track_ids"] = stable_track_ids
        stream["total_unique_persons"] = len(stable_track_ids)
    return streams, malformed_count


def _unique_track_ids(tracks: list[dict[str, Any]]) -> set[int]:
    track_ids: set[int] = set()
    for track in tracks:
        try:
            track_id = int(track.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if track_id >= 0:
            track_ids.add(track_id)
    return track_ids


def _missing_stream_ids(streams: dict[str, Any], expected_stream_count: int) -> list[str]:
    return [
        f"stream-{index}"
        for index in range(expected_stream_count)
        if f"stream-{index}" not in streams
    ]


def _file_sizes(paths: dict[str, Path]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for key, path in paths.items():
        if path.is_file():
            sizes[key] = path.stat().st_size
    return sizes


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "error": f"invalid JSON metadata: {path}"}


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote multifile summary: {output_path}")
    print(f"Observed streams: {summary['observed_stream_count']}/{summary['expected_stream_count']}")
    print(f"Total frames: {summary['total_frame_count']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Total unique persons: {summary['total_unique_persons']}")
    if summary["missing_stream_ids"]:
        print(f"Missing streams: {', '.join(summary['missing_stream_ids'])}")
    for stream_id, stream in summary["streams"].items():
        print(
            f"{stream_id}: frames={stream.get('frame_count', 0)} "
            f"detections={stream.get('total_detections', 0)} "
            f"unique={stream.get('total_unique_persons', 0)} "
            f"fps={stream.get('estimated_fps')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
