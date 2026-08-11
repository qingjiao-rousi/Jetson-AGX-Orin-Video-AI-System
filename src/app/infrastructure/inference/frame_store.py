from __future__ import annotations

from collections import deque
from threading import Lock
import time
from typing import Iterable

import cv2
import numpy as np

from app.domain.entities import canonical_stream_id


class FrameStore:
    """Bounded source-frame cache used by ROI tasks outside the probe thread."""

    def __init__(self, capture_stream_ids: Iterable[str] = (), max_size: int = 128) -> None:
        self._capture_stream_ids = {canonical_stream_id(value) for value in capture_stream_ids}
        self._max_size = max(int(max_size), 1)
        self._lock = Lock()
        self._order: deque[tuple[str, int]] = deque()
        self._frames: dict[tuple[str, int], tuple[np.ndarray, float, int]] = {}
        self._evicted = 0
        self._puts = 0
        self._pending_bytes = 0
        self._evicted_by_stream: dict[str, int] = {}
        self._consumer_hits: dict[str, int] = {}
        self._consumer_misses: dict[str, int] = {}
        self._consumer_frame_age_ms: dict[str, deque[float]] = {}

    def should_capture(self, stream_id: str) -> bool:
        return canonical_stream_id(stream_id) in self._capture_stream_ids

    def put(self, stream_id: str, frame_id: int, frame: np.ndarray) -> None:
        key = (canonical_stream_id(stream_id), int(frame_id))
        image = np.asarray(frame)
        if image.ndim != 3:
            return
        with self._lock:
            self._puts += 1
            if key in self._frames:
                previous = self._frames[key]
                copied = image.copy()
                self._pending_bytes += int(copied.nbytes) - previous[2]
                self._frames[key] = (copied, time.monotonic(), int(copied.nbytes))
                return
            while len(self._frames) >= self._max_size:
                old_key = self._order.popleft()
                previous = self._frames.pop(old_key, None)
                if previous is not None:
                    self._pending_bytes -= previous[2]
                    self._evicted += 1
                    self._evicted_by_stream[old_key[0]] = self._evicted_by_stream.get(old_key[0], 0) + 1
            copied = image.copy()
            self._frames[key] = (copied, time.monotonic(), int(copied.nbytes))
            self._pending_bytes += int(copied.nbytes)
            self._order.append(key)

    def get(self, stream_id: str, frame_id: int, *, consumer: str | None = None) -> np.ndarray | None:
        key = (canonical_stream_id(stream_id), int(frame_id))
        with self._lock:
            record = self._frames.get(key)
            if record is None:
                self._record_consumer_miss(consumer)
                return None
            self._record_consumer_hit(consumer, (time.monotonic() - record[1]) * 1000.0)
            return record[0].copy()

    def get_bgr(self, stream_id: str, frame_id: int, *, consumer: str | None = None) -> np.ndarray | None:
        frame = self.get(stream_id, frame_id, consumer=consumer)
        if frame is None:
            return None
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame

    def stats(self) -> dict[str, object]:
        with self._lock:
            consumers = set(self._consumer_hits) | set(self._consumer_misses) | set(self._consumer_frame_age_ms)
            return {
                "puts": self._puts,
                "pending_frames": len(self._frames),
                "pending_bytes": self._pending_bytes,
                "max_size": self._max_size,
                "evicted": self._evicted,
                "dropped": self._evicted,
                "evicted_by_stream": dict(sorted(self._evicted_by_stream.items())),
                "by_consumer": {
                    name: {
                        "hits": self._consumer_hits.get(name, 0),
                        "misses": self._consumer_misses.get(name, 0),
                        "frame_age_ms": _sample_summary(self._consumer_frame_age_ms.get(name, ())),
                    }
                    for name in sorted(consumers)
                },
            }

    def _record_consumer_hit(self, consumer: str | None, age_ms: float) -> None:
        if consumer is None:
            return
        self._consumer_hits[consumer] = self._consumer_hits.get(consumer, 0) + 1
        self._consumer_frame_age_ms.setdefault(consumer, deque(maxlen=2048)).append(max(age_ms, 0.0))

    def _record_consumer_miss(self, consumer: str | None) -> None:
        if consumer is None:
            return
        self._consumer_misses[consumer] = self._consumer_misses.get(consumer, 0) + 1


def _sample_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"samples": 0, "average": None, "p50": None, "p95": None, "max": None}
    return {
        "samples": len(ordered),
        "average": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "max": round(ordered[-1], 3),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
