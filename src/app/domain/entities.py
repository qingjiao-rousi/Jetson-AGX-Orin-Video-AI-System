from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
    bbox: BoundingBox


@dataclass(frozen=True)
class FrameResult:
    stream_id: str
    frame_id: int
    timestamp: datetime
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


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
