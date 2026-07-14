from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any


def canonical_stream_id(value: object, *, default_index: int = 0) -> str:
    """Normalize result stream IDs to zero-based ``stream-N`` values."""
    if value is None or isinstance(value, bool):
        return f"stream-{default_index}"
    if isinstance(value, int):
        return f"stream-{max(value, 0)}"

    text = str(value).strip()
    if not text:
        return f"stream-{default_index}"
    if text.isdigit():
        return f"stream-{max(int(text), 0)}"

    canonical = re.fullmatch(r"stream-(\d+)", text, flags=re.IGNORECASE)
    if canonical:
        return f"stream-{int(canonical.group(1))}"

    simulator_mount = re.fullmatch(r"stream(\d+)", text, flags=re.IGNORECASE)
    if simulator_mount:
        return f"stream-{max(int(simulator_mount.group(1)) - 1, 0)}"

    return text


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class Track:
    track_id: int
    class_id: int
    confidence: float
    bbox: BoundingBox
    global_track_id: int | None = None


@dataclass(frozen=True)
class FrameResult:
    stream_id: str
    frame_id: int
    timestamp: datetime
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    models: dict[str, "ModelResult"] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamStats:
    stream_id: str
    fps: float = 0.0
    latency_ms: float = 0.0
    dropped_frames: int = 0


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    frame_id: int
    stream_id: str
    timestamp: datetime
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    batch_id: int
    timestamp: datetime
    frame_results: list[FrameResult] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineState:
    is_running: bool = False
    source_count: int = 0
    last_error: str | None = None
