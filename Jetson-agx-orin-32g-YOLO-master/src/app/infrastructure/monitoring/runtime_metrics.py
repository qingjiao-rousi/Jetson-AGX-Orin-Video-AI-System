from __future__ import annotations

import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from app.domain.entities import FrameResult


class RuntimeMetricsRecorder:
    def __init__(
        self,
        path: Path | None,
        *,
        interval_seconds: float = 1.0,
        stale_after_seconds: float = 5.0,
        enable_last_frame_keepalive: bool = True,
        keepalive_timeout_ms: int = 1000,
    ) -> None:
        self._path = path
        self._interval_seconds = max(interval_seconds, 0.1)
        self._stale_after_seconds = max(stale_after_seconds, 0.1)
        self._enable_last_frame_keepalive = enable_last_frame_keepalive
        self._keepalive_timeout_seconds = max(keepalive_timeout_ms / 1000.0, 0.1)
        self._lock = Lock()
        self._file = None
        self._started_at = 0.0
        self._last_emit_at = 0.0
        self._total_frames = 0
        self._streams: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._last_emit_at = 0.0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")

    def observe(self, result: FrameResult, *, gpu_snapshot: dict[str, object] | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            self._total_frames += 1
            stream = self._streams.setdefault(
                result.stream_id,
                {
                    "stream_id": result.stream_id,
                    "frame_count": 0,
                    "first_seen_monotonic": now,
                    "last_seen_monotonic": now,
                    "last_frame_id": None,
                    "stale_count": 0,
                    "recovered_count": 0,
                    "status": "online",
                },
            )
            previous_status = stream.get("status")
            stream["frame_count"] += 1
            stream["last_seen_monotonic"] = now
            stream["last_frame_id"] = result.frame_id
            stream["last_seen_at"] = _utc_now()
            stream["last_keepalive_at"] = None
            stream["keepalive_active"] = False
            stream["detections"] = len(result.detections)
            stream["tracks"] = len(result.tracks)
            stream["status"] = "online"
            if previous_status == "stale":
                stream["recovered_count"] += 1

            if now - self._last_emit_at >= self._interval_seconds:
                self._mark_stale_streams(now)
                self._emit_locked(now, gpu_snapshot=gpu_snapshot)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._mark_stale_streams(now)
            return self._payload(now, gpu_snapshot=None)

    def close(self) -> None:
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.close()

    def _mark_stale_streams(self, now: float) -> None:
        for stream in self._streams.values():
            age = now - float(stream.get("last_seen_monotonic", now))
            if age <= self._stale_after_seconds:
                continue
            if stream.get("status") != "stale":
                stream["status"] = "stale"
                stream["stale_count"] = int(stream.get("stale_count", 0)) + 1
            stream["stale_seconds"] = round(age, 3)
            if self._enable_last_frame_keepalive and age >= self._keepalive_timeout_seconds:
                stream["keepalive_active"] = True
                stream["last_keepalive_at"] = _utc_now()

    def _emit_locked(self, now: float, *, gpu_snapshot: dict[str, object] | None) -> None:
        self._last_emit_at = now
        if self._file is None or self._file.closed:
            return
        self._file.write(json.dumps(self._payload(now, gpu_snapshot=gpu_snapshot), ensure_ascii=False) + "\n")
        self._file.flush()

    def _payload(self, now: float, *, gpu_snapshot: dict[str, object] | None) -> dict[str, Any]:
        elapsed = max(now - self._started_at, 0.001)
        return {
            "timestamp": _utc_now(),
            "pid": os.getpid(),
            "elapsed_seconds": round(elapsed, 3),
            "total_frames": self._total_frames,
            "processing_fps": round(self._total_frames / elapsed, 3),
            "process": _process_snapshot(),
            "gpu": gpu_snapshot or {},
            "streams": {
                stream_id: _stream_payload(stream, now)
                for stream_id, stream in sorted(self._streams.items())
            },
        }


def _stream_payload(stream: dict[str, Any], now: float) -> dict[str, Any]:
    first_seen = float(stream.get("first_seen_monotonic", now))
    last_seen = float(stream.get("last_seen_monotonic", now))
    elapsed = max(last_seen - first_seen, 0.001)
    age = now - last_seen
    frame_count = int(stream.get("frame_count", 0))
    last_frame_id = stream.get("last_frame_id")
    frame_span = max(int(last_frame_id) + 1, frame_count) if last_frame_id is not None else frame_count
    dropped_frames = max(frame_span - frame_count, 0)
    return {
        "stream_id": stream.get("stream_id"),
        "status": stream.get("status", "unknown"),
        "frame_count": frame_count,
        "last_frame_id": last_frame_id,
        "last_seen_at": stream.get("last_seen_at"),
        "last_keepalive_at": stream.get("last_keepalive_at"),
        "keepalive_active": bool(stream.get("keepalive_active", False)),
        "last_seen_age_seconds": round(age, 3),
        "frame_age_ms": round(max(age, 0.0) * 1000, 1),
        "estimated_processing_fps": round(int(stream.get("frame_count", 0)) / elapsed, 3),
        "detections": int(stream.get("detections", 0)),
        "detection_count": int(stream.get("detections", 0)),
        "tracks": int(stream.get("tracks", 0)),
        "track_count": int(stream.get("tracks", 0)),
        "dropped_frames": dropped_frames,
        "dropped_frame_rate": round(dropped_frames / max(frame_span, 1), 4),
        "stale_count": int(stream.get("stale_count", 0)),
        "recovered_count": int(stream.get("recovered_count", 0)),
    }


def _process_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_seconds": round(float(usage.ru_utime), 3),
        "system_cpu_seconds": round(float(usage.ru_stime), 3),
        "max_rss_kb": int(usage.ru_maxrss),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
