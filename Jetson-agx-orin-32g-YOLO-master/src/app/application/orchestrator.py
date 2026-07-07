from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from threading import Event
from typing import Any

from app.domain.entities import FrameResult


@dataclass
class Orchestrator:
    settings: object
    pipeline_manager: object
    meta_parser: object
    json_writer: object
    gpu_monitor: object
    fps_controller: object
    backpressure_controller: object

    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _last_result: FrameResult | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if self._started:
            return
        self.settings.validate()
        self._stop_event.clear()
        self.pipeline_manager.clear_error()
        self.pipeline_manager.probes().register_frame_result_handler(self.on_frame_result)
        self.gpu_monitor.start()
        self.pipeline_manager.start()
        self._started = True
        self._log_pipeline_summary()

    def run_forever(self) -> None:
        while not self._stop_event.wait(0.2):
            if hasattr(self.pipeline_manager, "running") and not self.pipeline_manager.running:
                self._stop_event.set()
                return

    def stop(self) -> None:
        if self._stop_event.is_set() and not self._started:
            return
        self._stop_event.set()
        self.pipeline_manager.stop()
        self.gpu_monitor.stop()
        self.json_writer.close()
        self._started = False

    def on_frame_result(self, result: FrameResult) -> None:
        if self._stop_event.is_set():
            return
        parsed = self.meta_parser.parse(result)
        self._last_result = parsed
        self.backpressure_controller.observe(parsed)
        self.fps_controller.observe(parsed)
        self.json_writer.write(parsed)

    def handle_error(self, message: str) -> None:
        self.pipeline_manager.set_error(message)
        logging.error("pipeline error: %s", message)
        self.stop()
        raise RuntimeError(message)

    def pipeline_state(self) -> dict[str, Any]:
        state = self.pipeline_manager.state()
        summary = self.pipeline_manager.describe()
        return {
            "is_running": state.is_running,
            "source_count": state.source_count,
            "last_error": state.last_error,
            "pipeline": summary,
            "last_result": self._result_summary(),
            "gpu_monitor_running": self.gpu_monitor.running,
        }

    def _log_pipeline_summary(self) -> None:
        summary = self.pipeline_manager.describe()
        logging.info(
            "pipeline started: sources=%s nodes=%s probes=%s",
            summary["source_count"],
            len(summary["nodes"]),
            len(summary["probes"]),
        )

    def _result_summary(self) -> dict[str, Any] | None:
        if self._last_result is None:
            return None
        return {
            "stream_id": self._last_result.stream_id,
            "frame_id": self._last_result.frame_id,
            "detection_count": len(self._last_result.detections),
            "track_count": len(self._last_result.tracks),
            "timestamp": self._last_result.timestamp.isoformat(),
        }

    def status_snapshot(self) -> dict[str, Any]:
        pipeline_state = self.pipeline_state()
        return {
            "app": {
                "started": self._started,
                "stop_requested": self._stop_event.is_set(),
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
            },
            "pipeline": pipeline_state.get("pipeline"),
            "pipeline_status": self.pipeline_manager.pipeline_status()
            if hasattr(self.pipeline_manager, "pipeline_status")
            else {},
            "bus": self.pipeline_manager.bus_state(),
            "writer": self.json_writer.stats() if hasattr(self.json_writer, "stats") else {},
            "monitor": self.gpu_monitor.snapshot() if hasattr(self.gpu_monitor, "snapshot") else {},
            "controllers": {
                "fps": self.fps_controller.stats() if hasattr(self.fps_controller, "stats") else {},
                "backpressure": self.backpressure_controller.stats()
                if hasattr(self.backpressure_controller, "stats")
                else {},
            },
            "last_result": pipeline_state.get("last_result"),
            "is_running": pipeline_state.get("is_running"),
            "source_count": pipeline_state.get("source_count"),
            "last_error": pipeline_state.get("last_error"),
        }
