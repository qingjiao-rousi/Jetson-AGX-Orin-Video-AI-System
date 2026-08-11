#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify one simulated RTSP publisher can recover after failure.")
    parser.add_argument("--runtime-dir", type=Path, default=Path(".runtime/mediamtx_sim"))
    parser.add_argument("--stream-id", default="stream1")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/rtsp_recovery_check.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = check_recovery(
        args.runtime_dir,
        stream_id=args.stream_id,
        timeout=args.timeout,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_report(report, args.output_json)
    return 0 if report["recovered"] else 1


def check_recovery(runtime_dir: Path, *, stream_id: str, timeout: float) -> dict[str, Any]:
    status_path = runtime_dir / "source_status.json"
    before = _read_status(status_path)
    stream = _find_stream(before, stream_id)
    started_at = _now()
    if not stream:
        return _report(started_at, stream_id, before, {}, False, "stream not found")
    pid = stream.get("pid")
    restart_count = int(stream.get("restart_count", 0))
    if not isinstance(pid, int) or pid <= 0:
        return _report(started_at, stream_id, before, {}, False, f"stream pid is invalid: {pid}")

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return _report(started_at, stream_id, before, {}, False, f"failed to kill pid {pid}: {exc}")

    deadline = time.monotonic() + timeout
    after: dict[str, Any] = {}
    while time.monotonic() < deadline:
        time.sleep(1.0)
        after = _read_status(status_path)
        current = _find_stream(after, stream_id)
        if not current:
            continue
        new_pid = current.get("pid")
        new_restart_count = int(current.get("restart_count", 0))
        if (
            current.get("status") == "online"
            and isinstance(new_pid, int)
            and new_pid > 0
            and new_pid != pid
            and new_restart_count > restart_count
        ):
            return _report(started_at, stream_id, before, after, True, "")

    return _report(started_at, stream_id, before, after, False, "stream did not recover before timeout")


def _report(
    started_at: str,
    stream_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    recovered: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "finished_at": _now(),
        "stream_id": stream_id,
        "recovered": recovered,
        "error": error,
        "before": _stream_snapshot(before, stream_id),
        "after": _stream_snapshot(after, stream_id),
        "overall_after": {
            "status": after.get("status"),
            "online_count": after.get("online_count"),
            "stream_count": after.get("stream_count"),
        }
        if after
        else {},
    }


def _read_status(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _find_stream(status: dict[str, Any], stream_id: str) -> dict[str, Any]:
    for stream in status.get("streams", []):
        if stream.get("stream_id") == stream_id:
            return stream
    return {}


def _stream_snapshot(status: dict[str, Any], stream_id: str) -> dict[str, Any]:
    stream = _find_stream(status, stream_id)
    return {
        "status": stream.get("status"),
        "pid": stream.get("pid"),
        "restart_count": stream.get("restart_count"),
        "consecutive_failures": stream.get("consecutive_failures"),
        "last_error": stream.get("last_error"),
        "uri": stream.get("uri"),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def print_report(report: dict[str, Any], output_path: Path) -> None:
    print(f"Wrote RTSP recovery report: {output_path}")
    print(f"Stream: {report['stream_id']}")
    print(f"Recovered: {report['recovered']}")
    if report.get("error"):
        print(f"Error: {report['error']}")
    print(f"Before: {report['before']}")
    print(f"After: {report['after']}")


if __name__ == "__main__":
    raise SystemExit(main())
