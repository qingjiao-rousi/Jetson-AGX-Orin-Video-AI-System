#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check one in-process RTSP pipeline output summary.")
    parser.add_argument("summary_json", type=Path, help="Input rtsp_summary.json path.")
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        help="Output quality JSON path. Defaults to rtsp_quality.json beside summary.",
    )
    parser.add_argument("--min-fps", type=float, default=1.0)
    parser.add_argument("--min-metric-samples", type=int, default=1)
    parser.add_argument("--max-stale-count", type=int, default=5)
    parser.add_argument("--require-person", action="store_true")
    parser.add_argument("--strict-frame-continuity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = args.output_json or (args.summary_json.parent / "rtsp_quality.json")
    quality = check_rtsp(
        args.summary_json,
        min_fps=args.min_fps,
        min_metric_samples=args.min_metric_samples,
        max_stale_count=args.max_stale_count,
        require_person=args.require_person,
        strict_frame_continuity=args.strict_frame_continuity,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_quality(quality, output_json)
    return 1 if quality["quality_status"] == "failed" else 0


def check_rtsp(
    summary_json: Path,
    *,
    min_fps: float = 1.0,
    min_metric_samples: int = 1,
    max_stale_count: int = 5,
    require_person: bool = False,
    strict_frame_continuity: bool = False,
) -> dict[str, Any]:
    summary = _read_json(summary_json)
    output_dir = summary_json.parent
    failures: list[str] = []
    reviews: list[str] = []

    _check_file(summary.get("results_jsonl"), "results JSONL", output_dir, failures)
    _check_file(summary.get("metrics_jsonl"), "runtime metrics JSONL", output_dir, failures)
    metrics_stats = _read_jsonl_stats(summary.get("metrics_jsonl"))
    metrics_tail = metrics_stats.get("tail", {})
    if int(metrics_stats.get("line_count", 0)) < min_metric_samples:
        failures.append(
            f"runtime metrics samples {metrics_stats.get('line_count', 0)} < {min_metric_samples}"
        )
    gpu_status = str((metrics_tail.get("gpu") or {}).get("status", "unknown"))
    if gpu_status in ("unavailable", "unknown", ""):
        reviews.append(f"tegrastats GPU metrics unavailable: {(metrics_tail.get('gpu') or {}).get('reason', gpu_status)}")
    for stream_id, stream in sorted((metrics_tail.get("streams") or {}).items()):
        status = stream.get("status")
        stale_count = int(stream.get("stale_count", 0))
        if status == "stale" and stale_count > max_stale_count:
            reviews.append(
                f"{stream_id}: stale for {stream.get('last_seen_age_seconds')}s"
            )
        if stale_count > max_stale_count:
            reviews.append(f"{stream_id}: stale_count={stream.get('stale_count')}")
        if stream.get("keepalive_active") is True and stale_count > max_stale_count:
            reviews.append(f"{stream_id}: last-frame keepalive is active")
    tiled_video = summary.get("tiled_video")
    if str(summary.get("output_sink", "")).lower() == "file":
        _check_file(tiled_video, "RTSP preview MP4", output_dir, failures)
    _check_file(summary.get("run_log"), "run.log", output_dir, failures)
    _check_file(summary.get("run_metadata"), "run metadata", output_dir, failures)
    _check_file(summary.get("source_status"), "source status", output_dir, failures)

    source_status = summary.get("source_status_payload") or {}
    if source_status:
        if source_status.get("status") != "online":
            failures.append(f"source simulator status is {source_status.get('status')}")
        online = int(source_status.get("online_count", 0))
        stream_count = int(source_status.get("stream_count", 0))
        if stream_count and online != stream_count:
            failures.append(f"source simulator online count {online} != stream count {stream_count}")
    recovery_check = _read_json_if_exists(summary.get("recovery_check"))
    if recovery_check and recovery_check.get("recovered") is not True:
        failures.append(f"RTSP recovery check failed: {recovery_check.get('error') or 'not recovered'}")

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
        stream_quality = _check_stream(
            str(stream_id),
            stream,
            min_fps=min_fps,
            require_person=require_person,
            strict_frame_continuity=strict_frame_continuity,
        )
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
        "min_metric_samples": min_metric_samples,
        "max_stale_count": max_stale_count,
        "require_person": require_person,
        "rule_definition": _rule_definition(
            min_fps=min_fps,
            min_metric_samples=min_metric_samples,
            max_stale_count=max_stale_count,
            require_person=require_person,
            strict_frame_continuity=strict_frame_continuity,
        ),
        "metrics": {
            "line_count": int(metrics_stats.get("line_count", 0)),
            "last_gpu": metrics_tail.get("gpu") or {},
            "last_processing_fps": metrics_tail.get("processing_fps"),
        },
        "failures": failures,
        "reviews": reviews,
        "malformed_json_line_count": malformed_json_line_count,
        "streams": stream_items,
    }


def _check_stream(
    stream_id: str,
    stream: dict[str, Any],
    *,
    min_fps: float,
    require_person: bool,
    strict_frame_continuity: bool,
) -> dict[str, Any]:
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
    if require_person and detections <= 0:
        reviews.append("zero detections")
    if require_person and track_observations <= 0:
        reviews.append("zero track observations")

    estimated_fps = _to_float(stream.get("estimated_fps"))
    if estimated_fps is None:
        reviews.append("missing estimated FPS")
    elif estimated_fps < min_fps:
        reviews.append(f"estimated FPS {estimated_fps:.2f} < {min_fps:.2f}")

    if strict_frame_continuity and stream.get("is_frame_continuous") is False:
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


def _read_json_if_exists(raw_path: Any) -> dict[str, Any]:
    if not raw_path:
        return {}
    path = Path(str(raw_path))
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_jsonl_tail(raw_path: Any) -> dict[str, Any]:
    if not raw_path:
        return {}
    path = Path(str(raw_path))
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {}
    return {}


def _read_jsonl_stats(raw_path: Any) -> dict[str, Any]:
    if not raw_path:
        return {"line_count": 0, "tail": {}}
    path = Path(str(raw_path))
    if not path.is_file():
        return {"line_count": 0, "tail": {}}
    line_count = 0
    tail: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"line_count": 0, "tail": {}}
    for line in lines:
        if not line.strip():
            continue
        line_count += 1
        try:
            tail = json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"line_count": line_count, "tail": tail}


def _rule_definition(
    *,
    min_fps: float,
    min_metric_samples: int,
    max_stale_count: int,
    require_person: bool,
    strict_frame_continuity: bool,
) -> dict[str, Any]:
    return {
        "passed": [
            "pipeline run_status is ok and exit_code is 0",
            "results.jsonl, runtime_metrics.jsonl, run.log, run_metadata.json and source_status.json exist and are non-empty",
            "source simulator status is online and online_count equals stream_count",
            "observed stream count equals expected stream count",
            "results JSONL has valid rows and no malformed JSON rows",
            f"each stream has frame_count > 0 and estimated_fps >= {min_fps}",
            f"runtime_metrics.jsonl contains at least {min_metric_samples} samples",
            f"each stream stale_count <= {max_stale_count}",
            "run.log contains no fatal patterns",
        ],
        "review": [
            "zero detections or zero track observations when require_person is enabled",
            "timestamps are not monotonic",
            "frame ids are not continuous when strict_frame_continuity is enabled",
            "tegrastats GPU metrics are unavailable",
            "non-fatal warnings appear in run.log",
            "last-frame keepalive is active",
            "require_person is enabled but no stable person track is found",
        ],
        "failed": [
            "pipeline exits non-zero or run_status is not ok",
            "required files are missing or empty",
            "observed stream count is lower than expected",
            "source simulator is offline or partially online",
            "results JSONL has no rows or malformed rows",
            "total frame count is zero",
            "RTSP recovery check fails",
            "run.log contains fatal patterns such as Traceback or failed PLAYING",
        ],
        "require_person": require_person,
        "strict_frame_continuity": strict_frame_continuity,
    }


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
    benign_warning_patterns = (
        "does not support property `custom-lib-path`",
        "does not support property `enable-batch-process`",
        "Implicit layer support has been deprecated",
        "Using an engine plan file across different models of devices is not recommended",
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
            if any(pattern in line for pattern in benign_warning_patterns):
                continue
            if any(pattern in line for pattern in review_patterns):
                reviews.append(f"run log warning: {line.strip()[:240]}")
                if len(reviews) >= 5:
                    break
    return {"failures": failures, "reviews": reviews}


def print_quality(quality: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote RTSP quality: {output_path}")
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
