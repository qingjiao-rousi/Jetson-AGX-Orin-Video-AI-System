from __future__ import annotations

"""跨 pipeline、路由、worker 与输出模块共享的不可变领域对象。"""

# frozen 使一帧结果成为只读快照，异步 worker 只能通过请求/事件沟通，不能回写主结果。
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any


def canonical_stream_id(value: object, *, default_index: int = 0) -> str:
    """将各输入形式统一为从零开始的 ``stream-N``。

    DeepStream 使用 ``pad_index``，RTSP 模拟器可能使用 ``stream1``，而 JSON/C++ 路径
    常已是 ``stream-0``；统一命名是 per-stream 指标、FrameStore 与路由能对齐的前提。
    """
    # bool 是 int 子类，必须先排除，否则 True 会被错误规范为 stream-1。
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
    """左上角加宽高的像素坐标框，与 DeepStream ``rect_params`` 保持一致。"""
    left: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class Detection:
    """单帧检测，不要求 tracker 已分配 ID。"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class Track:
    """检测关联后的轨迹；track_id 为本流本地 ID，global_track_id 保留 tracker 原始 ID。"""
    track_id: int
    class_id: int
    confidence: float
    bbox: BoundingBox
    global_track_id: int | None = None
    class_name: str = "unknown"


@dataclass(frozen=True)
class FrameResult:
    """应用层的主帧结果快照，是路由、指标、JSONL 和场景分析共同输入。"""
    stream_id: str
    frame_id: int
    timestamp: datetime
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    models: dict[str, "ModelResult"] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamStats:
    """面向单路状态展示的轻量统计，不替代运行时完整 metrics。"""
    stream_id: str
    fps: float = 0.0
    latency_ms: float = 0.0
    dropped_frames: int = 0


@dataclass(frozen=True)
class ModelResult:
    """专用模型关联到源帧后的结果容器。"""
    model_name: str
    frame_id: int
    stream_id: str
    timestamp: datetime
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    """批处理边界对象；主业务逻辑仍以 FrameResult 为粒度。"""
    batch_id: int
    timestamp: datetime
    frame_results: list[FrameResult] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineState:
    """PipelineManager 暴露给编排器和 dashboard 的最小生命周期状态。"""
    is_running: bool = False
    source_count: int = 0
    last_error: str | None = None
