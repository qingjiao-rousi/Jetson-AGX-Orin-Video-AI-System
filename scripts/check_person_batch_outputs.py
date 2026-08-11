#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check batch person analytics outputs.")
    parser.add_argument("batch_summary_json", type=Path, help="Input batch_summary.json path.")
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        help="Output quality JSON path. Defaults to batch_quality.json beside batch_summary.json.",
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        default=1.0,
        help="Minimum estimated FPS before a video is marked for review.",
    )
    parser.add_argument(
        "--require-person",
        action="store_true",
        help="Mark videos with zero unique persons for review.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = args.output_json or (args.batch_summary_json.parent / "batch_quality.json")
    quality = check_batch(args.batch_summary_json, min_fps=args.min_fps, require_person=args.require_person)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_quality(quality, output_json)
    return 1 if quality["failed_count"] else 0


def check_batch(batch_summary_json: Path, min_fps: float = 1.0, require_person: bool = False) -> dict[str, Any]:
    summary = _read_json(batch_summary_json)
    batch_dir = batch_summary_json.parent
    videos = []
    passed_count = 0
    review_count = 0
    failed_count = 0

    for index, video in enumerate(summary.get("videos", []), start=1):
        item = check_video(video, index=index, batch_dir=batch_dir, min_fps=min_fps, require_person=require_person)
        videos.append(item)
        if item["quality_status"] == "passed":
            passed_count += 1
        elif item["quality_status"] == "review":
            review_count += 1
        else:
            failed_count += 1

    return {
        "batch_summary_json": str(batch_summary_json),
        "video_count": len(videos),
        "passed_count": passed_count,
        "review_count": review_count,
        "failed_count": failed_count,
        "min_fps": min_fps,
        "require_person": require_person,
        "videos": videos,
    }


def check_video(
    video: dict[str, Any],
    index: int,
    batch_dir: Path,
    min_fps: float,
    require_person: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    reviews: list[str] = []

    status = str(video.get("status", "unknown"))
    if status != "ok":
        failures.append(f"run status is {status}")
        if video.get("error"):
            failures.append(str(video["error"]))

    _check_existing_file(video.get("output_video"), "output video", batch_dir, failures)
    _check_existing_file(video.get("output_overlay_video"), "overlay video", batch_dir, failures)
    _check_existing_file(video.get("output_jsonl"), "output JSONL", batch_dir, failures)
    _check_existing_file(video.get("output_summary"), "analytics summary", batch_dir, failures)

    total_unique_persons = _to_int(video.get("total_unique_persons"))
    if require_person and total_unique_persons <= 0:
        reviews.append("no unique persons found")

    if "total_unique_persons" in video and total_unique_persons <= 0:
        reviews.append("zero unique persons")

    streams = video.get("streams", {})
    if not isinstance(streams, dict) or not streams:
        reviews.append("missing timeline stream summary")
    else:
        for stream_id, stream in sorted(streams.items()):
            if not isinstance(stream, dict):
                reviews.append(f"{stream_id}: invalid stream summary")
                continue
            if not stream.get("is_frame_continuous", False):
                reviews.append(f"{stream_id}: frame ids are not continuous")
            frame_count = _to_int(stream.get("frame_count"))
            if frame_count <= 0:
                reviews.append(f"{stream_id}: frame count is zero")
            estimated_fps = _to_float(stream.get("estimated_fps"))
            if estimated_fps is None:
                reviews.append(f"{stream_id}: missing estimated FPS")
            elif estimated_fps < min_fps:
                reviews.append(f"{stream_id}: estimated FPS {estimated_fps:.2f} < {min_fps:.2f}")

    log_findings = _scan_log_for_failures(video.get("log_path"))
    failures.extend(log_findings["failures"])
    reviews.extend(log_findings["reviews"])

    quality_status = "failed" if failures else "review" if reviews else "passed"
    return {
        "index": index,
        "input_video": video.get("input_video", ""),
        "quality_status": quality_status,
        "run_status": status,
        "failures": failures,
        "reviews": reviews,
        "total_unique_persons": video.get("total_unique_persons", 0),
        "line_crossing_in": video.get("line_crossing_in", 0),
        "line_crossing_out": video.get("line_crossing_out", 0),
        "output_dir": video.get("output_dir", ""),
    }


def _check_existing_file(raw_path: Any, label: str, batch_dir: Path, failures: list[str]) -> None:
    if not raw_path:
        failures.append(f"{label} path is missing")
        return
    path = _resolve_output_path(raw_path, batch_dir)
    if not path.is_file():
        failures.append(f"{label} not found: {raw_path}")
        return
    if path.stat().st_size <= 0:
        failures.append(f"{label} is empty: {raw_path}")


def _resolve_output_path(raw_path: Any, batch_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    candidates = (
        path,
        batch_dir / path,
        batch_dir.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_log_for_failures(raw_path: Any) -> dict[str, list[str]]:
    if not raw_path:
        return {"failures": [], "reviews": []}
    path = Path(str(raw_path))
    if not path.is_file():
        return {"failures": [], "reviews": []}
    fatal_patterns = (
        "Traceback",
        "application crashed",
        "RuntimeError:",
        "failed to set GStreamer pipeline to PLAYING",
    )
    review_patterns = (
        "WARNING",
        "warning",
    )
    failures: list[str] = []
    reviews: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"failures": [], "reviews": []}
    for line in lines:
        if any(pattern in line for pattern in fatal_patterns):
            failures.append(f"run log fatal: {line.strip()[:240]}")
            if len(failures) >= 5:
                break
    if not failures:
        for line in lines:
            if any(pattern in line for pattern in review_patterns):
                reviews.append(f"run log warning: {line.strip()[:240]}")
                if len(reviews) >= 5:
                    break
    return {"failures": failures, "reviews": reviews}


def print_quality(quality: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote batch quality: {output_path}")
    print(f"Videos: {quality['video_count']}")
    print(f"Passed: {quality['passed_count']}")
    print(f"Need review: {quality['review_count']}")
    print(f"Failed: {quality['failed_count']}")
    for video in quality["videos"]:
        name = Path(str(video.get("input_video", ""))).name
        status = str(video["quality_status"]).upper()
        messages = video["failures"] or video["reviews"]
        suffix = f": {'; '.join(messages)}" if messages else ""
        print(f"[{status}] {name}{suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
