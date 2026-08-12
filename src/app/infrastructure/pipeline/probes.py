from __future__ import annotations

"""把 pipeline probe payload 转发为 ``FrameResult`` 的轻量回调注册表。"""

from dataclasses import dataclass, field
from typing import Callable, Optional

# Registry 只认识领域结果，因而不依赖 Gst/pyds，可直接用于单测模拟 probe 输入。
from app.domain.entities import FrameResult


FrameResultHandler = Callable[[FrameResult], None]


@dataclass
class ProbeRegistry:
    """连接 GStreamer probe 与应用编排层。

    真实 pad probe 由 ``PipelineBuilder`` 挂载；本类故意不依赖 Gst/pyds，只处理
    payload -> ``MetaParser`` -> 回调的转换，因此可在普通 Python 单测中验证。
    """

    frame_result_handler: Optional[FrameResultHandler] = None
    _events: list[str] = field(default_factory=list, init=False, repr=False)
    _event_limit: int = field(default=100, init=False, repr=False)

    def register_frame_result_handler(self, handler: FrameResultHandler) -> None:
        """注册唯一的应用入口。当前架构只允许一个编排器消费主检测帧结果。"""
        self.frame_result_handler = handler
        self._record_event("frame_result_handler_registered")

    def emit_frame_result(self, result: FrameResult) -> None:
        """同步交给编排器；耗时专用推理由编排器提交给异步 worker。"""
        if self.frame_result_handler is None:
            return
        self.frame_result_handler(result)

    def emit_probe_payload(self, payload: object, parser) -> None:
        """解析一个 batch 中的全部帧，逐帧发射以保持下游的 per-stream/per-frame 语义。"""
        # 一个 nvstreammux batch 可能包含多路帧，parse_many 后必须逐帧交给 Orchestrator。
        results = (
            parser.parse_many(payload) if hasattr(parser, "parse_many") else (parser.parse(payload),)
        )
        for result in results:
            self._record_event("frame_result_emitted")
            self.emit_frame_result(result)

    def events(self) -> tuple[str, ...]:
        return tuple(self._events)

    def _record_event(self, event: str) -> None:
        """保留最近事件供 debug API 检查回调是否接通，不作为持久化审计日志。"""
        self._events.append(event)
        if len(self._events) > self._event_limit:
            del self._events[: len(self._events) - self._event_limit]
