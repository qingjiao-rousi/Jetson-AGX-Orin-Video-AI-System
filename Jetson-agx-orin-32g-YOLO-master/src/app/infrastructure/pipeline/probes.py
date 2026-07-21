from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from app.domain.entities import FrameResult


FrameResultHandler = Callable[[FrameResult], None]


@dataclass
class ProbeRegistry:
    """
    Lightweight probe registry for the current single-service architecture.

    In the initial version this keeps callback wiring explicit and simple.
    Later it can be replaced by actual DeepStream pad-probe hooks.
    """

    frame_result_handler: Optional[FrameResultHandler] = None
    _events: list[str] = field(default_factory=list, init=False, repr=False)
    _event_limit: int = field(default=100, init=False, repr=False)

    def register_frame_result_handler(self, handler: FrameResultHandler) -> None:
        self.frame_result_handler = handler
        self._record_event("frame_result_handler_registered")

    def emit_frame_result(self, result: FrameResult) -> None:
        if self.frame_result_handler is None:
            return
        self.frame_result_handler(result)

    def emit_probe_payload(self, payload: object, parser) -> None:
        results = (
            parser.parse_many(payload) if hasattr(parser, "parse_many") else (parser.parse(payload),)
        )
        for result in results:
            self._record_event("frame_result_emitted")
            self.emit_frame_result(result)

    def events(self) -> tuple[str, ...]:
        return tuple(self._events)

    def _record_event(self, event: str) -> None:
        self._events.append(event)
        if len(self._events) > self._event_limit:
            del self._events[: len(self._events) - self._event_limit]
