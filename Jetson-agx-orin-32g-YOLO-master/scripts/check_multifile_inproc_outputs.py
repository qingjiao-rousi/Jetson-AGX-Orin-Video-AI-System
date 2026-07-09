#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check one in-process multi-file pipeline output summary.")
    parser.add_argument("summary_json", type=Path, help="Input multifile_summary.json path.")
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        help="Output quality JSON path. Defaults to multifile_quality.json beside summary.",
    )
    parser.add_argument("--min-fps", type=float, default=1.0)
    parser.add_argument("--require-person", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = args.output_json or (args.summary_json.parent / "multifile_quality.json")
    quality = check_multifile(args.summary_json, min_fps=args.min_fps, require_person=args.require_person)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_quality(quality, output_json)
    return 1 if quality["quality_status"] == "failed" else 0


def check_multifile(summary_json: Path, *, min_fps: float = 1.0, require_person: bool = False) -> dict[str, Any]:
    summary = _read_json(summary_json)
    output_dir = summary_json.parent
    failures: list[str] = []
    reviews: list[str] = []

    _check_file(summary.get("results_jsonl"), "results JSONL", output_dir, failures)
    _check_file(summary.get("tiled_video"), "tiled MP4", output_dir, failures)
    _check_file(summary.get("run_log"), "run.log", output_dir, failures)
    _check_file(summary.get("run_metadata"), "run metadata", output_dir, failures)

    run_status = str(summary.get("run_status", "unknown"))
    if run_status != "ok":
        failures.append(f"run status is {run_status}")
        if summary.get("error"):
            failures.append(str(summary["error"]))
    exit_code = summary.get("exit_code")
    if exit_code not in (None, 0):
        failures.append(f"pipeline exited with code {exit_code}")

    expected = int(summary.get("expected_stream_count", 0))
    observed = int(summary.get("observed_stream_count", 0))
    if expected > 0 and observed != expected:
        failures.append(f"observed stream count {observed} != expected {expected}")

    missing_stream_ids = list(summary.get("missing_stream_ids", []))
    if missing_stream_ids:
        failures.append(f"missing streams: {', '.join(str(item) for item in missing_stream_ids)}")

    if int(summary.get("total_lines", 0)) <= 0:
        failures.append("results JSONL has no rows")
    malformed_json_line_count = int(summary.get("malformed_json_line_count", 0))
    if malformed_json_line_count > 0:
        failures.append(f"results JSONL has {malformed_json_line_count} malformed rows")
    if int(summary.get("total_frame_count", 0)) <= 0:
        failures.append("total frame count is zero")
    if require_person and int(summary.get("total_unique_persons", 0)) <= 0:
        reviews.append("no unique persons found")

    stream_items = []
    for stream_id, stream in sorted((summary.get("streams") or {}).items()):
        stream_quality = _check_stream(str(stream_id), stream, min_fps=min_fps, require_person=require_person)
        stream_items.append(stream_quality)
        reviews.extend(f"{stream_id}: {message}" for message in stream_quality["reviews"])
        failures.extend(f"{stream_id}: {message}" for message in stream_quality["failures"])

    log_findings = _scan_log_for_failures(summary.get("run_log"))
    failures.extend(log_findings["failures"])
    reviews.extend(log_findings["reviews"])

    status = "failed" if failures else "review" if reviews else "passed"
    return {
        "summary_json": str(summary_json),
        "output_dir": str(output_dir),
        "quality_status": status,
        "expected_stream_count": expected,
        "observed_stream_count": observed,
        "passed_stream_count": len([item for item in stream_items if item["quality_status"] == "passed"]),
        "review_stream_count": len([item for item in stream_items if item["quality_status"] == "review"]),
        "failed_stream_count": len([item for item in stream_items if item["quality_status"] == "failed"]),
        "min_fps": min_fps,
        "require_person": require_person,
        "failures": failures,
        "reviews": reviews,
        "malformed_json_line_count": malformed_json_line_count,
        "streams": stream_items,
    }


def _check_stream(stream_id: str, stream: dict[str, Any], *, min_fps: float, require_person: bool) -> dict[str, Any]:
    failures: list[str] = []
    reviews: list[str] = []

    frame_count = _to_int(stream.get("frame_count"))
    if frame_count <= 0:
        failures.append("frame count is zero")

    detections = _to_int(stream.get("total_detections"))
    track_observations = _to_int(stream.get("total_track_observations"))
    unique_persons = _to_int(stream.get("total_unique_persons"))
    if require_person and unique_persons <= 0:
        reviews.append("no unique persons found")
    if detections <= 0:
        reviews.append("zero detections")
    if track_observations <= 0:
        reviews.append("zero track observations")

    estimated_fps = _to_float(stream.get("estimated_fps"))
    if estimated_fps is None:
        reviews.append("missing estimated FPS")
    elif estimated_fps < min_fps:
        reviews.append(f"estimated FPS {estimated_fps:.2f} < {min_fps:.2f}")

    if stream.get("is_frame_continuous") is False:
        reviews.append("frame ids are not continuous")
    if stream.get("is_timestamp_monotonic") is False:
        reviews.append("timestamps are not monotonic")

    status = "failed" if failures else "review" if reviews else "passed"
    return {
        "stream_id": stream_id,
        "quality_status": status,
        "failures": failures,
        "reviews": reviews,
        "frame_count": frame_count,
        "total_detections": detections,
        "total_track_observations": track_observations,
        "total_unique_persons": unique_persons,
        "estimated_fps": estimated_fps,
    }


def _check_file(raw_path: Any, label: str, output_dir: Path, failures: list[str]) -> None:
    if not raw_path:
        failures.append(f"{label} path is missing")
        return
    path = Path(str(raw_path))
    if not path.is_absolute():
        candidates = (path, output_dir / path, output_dir.parent / path)
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    if not path.is_file():
        failures.append(f"{label} not found: {raw_path}")
        return
    if path.stat().st_size <= 0:
        failures.append(f"{label} is empty: {raw_path}")


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
    print(f"Wrote multifile quality: {output_path}")
    print(f"Status: {quality['quality_status']}")
    print(f"Streams: {quality['observed_stream_count']}/{quality['expected_stream_count']}")
    print(
        "Stream status: "
        f"passed={quality['passed_stream_count']} "
        f"review={quality['review_stream_count']} "
        f"failed={quality['failed_stream_count']}"
    )
    messages = quality["failures"] or quality["reviews"]
    for message in messages[:20]:
        print(f"- {message}")


if __name__ == "__main__":
    raise SystemExit(main())
