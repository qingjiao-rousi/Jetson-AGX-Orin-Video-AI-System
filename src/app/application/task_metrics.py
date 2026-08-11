from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Iterable


def sample_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    """Summarize a bounded latency sample without retaining unbounded history."""
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


class TaskExecutionMetrics:
    """Bounded execution metrics shared by non-primary task workers."""

    def __init__(self, sample_limit: int = 2048) -> None:
        self.processed = 0
        self.missing_frames = 0
        self.errors = 0
        self._lock = Lock()
        self._queue_wait_ms: deque[float] = deque(maxlen=sample_limit)
        self._inference_ms: deque[float] = deque(maxlen=sample_limit)
        self._task_latency_ms: deque[float] = deque(maxlen=sample_limit)

    def record_queue_wait(self, value_ms: float) -> None:
        with self._lock:
            self._queue_wait_ms.append(max(float(value_ms), 0.0))

    def record_inference(self, value_ms: float) -> None:
        with self._lock:
            self._inference_ms.append(max(float(value_ms), 0.0))

    def record_task_latency(self, value_ms: float) -> None:
        with self._lock:
            self._task_latency_ms.append(max(float(value_ms), 0.0))

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "processed": self.processed,
                "missing_frames": self.missing_frames,
                "errors": self.errors,
                "queue_wait_ms": sample_summary(tuple(self._queue_wait_ms)),
                "inference_ms": sample_summary(tuple(self._inference_ms)),
                "task_latency_ms": sample_summary(tuple(self._task_latency_ms)),
            }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
