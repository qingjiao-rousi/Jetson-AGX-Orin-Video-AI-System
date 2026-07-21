from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from threading import Event
import time
from typing import Any

from app.domain.entities import FrameResult
from app.application.routing_policy import RoutingPolicy, TaskRequestBuffer
from app.application.helmet_service import HelmetEvent
from app.application.plate_service import VehiclePassEvent
from app.application.pose_service import PoseEvent
from app.application.fire_smoke_service import FireSmokeEvent


@dataclass
class Orchestrator:
    settings: object
    pipeline_manager: object
    meta_parser: object
    json_writer: object
    gpu_monitor: object
    fps_controller: object
    backpressure_controller: object
    runtime_metrics: object | None = None
    routing_policy: RoutingPolicy | None = None
    task_buffer: TaskRequestBuffer | None = None
    helmet_worker: object | None = None
    plate_worker: object | None = None
    event_writer: object | None = None
    scene_analytics: object | None = None
    pose_worker: object | None = None
    fire_smoke_worker: object | None = None

    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _last_result: FrameResult | None = field(default=None, init=False, repr=False)
    _logged_routing_tasks: set[str] = field(default_factory=set, init=False, repr=False)

    def start(self) -> None:
        if self._started:
            return
        self.settings.validate()
        self._stop_event.clear()
        self.pipeline_manager.clear_error()
        self.pipeline_manager.probes().register_frame_result_handler(self.on_frame_result)
        if hasattr(self.pipeline_manager, "set_frame_gate") and hasattr(self.fps_controller, "should_drop_frame"):
            self.pipeline_manager.set_frame_gate(self.fps_controller.should_drop_frame)
        self.gpu_monitor.start()
        if hasattr(self.runtime_metrics, "start"):
            self.runtime_metrics.start()
        self.pipeline_manager.start()
        if self.helmet_worker is not None and hasattr(self.helmet_worker, "start"):
            self.helmet_worker.start()
        if self.plate_worker is not None and hasattr(self.plate_worker, "start"):
            self.plate_worker.start()
        if self.pose_worker is not None and hasattr(self.pose_worker, "start"):
            self.pose_worker.start()
        if self.fire_smoke_worker is not None and hasattr(self.fire_smoke_worker, "start"):
            self.fire_smoke_worker.start()
        self._started = True
        self._log_pipeline_summary()

    def run_forever(self, max_runtime_seconds: float | None = None) -> None:
        deadline = None
        if max_runtime_seconds is not None and max_runtime_seconds > 0:
            deadline = time.monotonic() + max_runtime_seconds
        while not self._stop_event.wait(0.2):
            if deadline is not None and time.monotonic() >= deadline:
                self._stop_event.set()
                return
            if hasattr(self.pipeline_manager, "running") and not self.pipeline_manager.running:
                if deadline is None:
                    self._stop_event.set()
                    return

    def stop(self) -> None:
        if self._stop_event.is_set() and not self._started:
            return
        self._stop_event.set()
        if self.helmet_worker is not None and hasattr(self.helmet_worker, "stop"):
            self.helmet_worker.stop()
        if self.plate_worker is not None and hasattr(self.plate_worker, "stop"):
            self.plate_worker.stop()
        if self.pose_worker is not None and hasattr(self.pose_worker, "stop"):
            self.pose_worker.stop()
        if self.fire_smoke_worker is not None and hasattr(self.fire_smoke_worker, "stop"):
            self.fire_smoke_worker.stop()
        self.pipeline_manager.stop()
        self.gpu_monitor.stop()
        if hasattr(self.runtime_metrics, "close"):
            self.runtime_metrics.close()
        if self.event_writer is not None and hasattr(self.event_writer, "close"):
            self.event_writer.close()
        self.json_writer.close()
        self._started = False

    def on_frame_result(self, result: FrameResult) -> None:
        if self._stop_event.is_set():
            return
        try:
            self._last_result = result
            self.backpressure_controller.observe(result)
            if self.scene_analytics is not None:
                for event in self.scene_analytics.observe(result):
                    if self.event_writer is not None and hasattr(self.event_writer, "write"):
                        self.event_writer.write(event)
            if self.routing_policy is not None:
                requests = self.routing_policy.route(result)
                if self.task_buffer is not None:
                    self.task_buffer.submit(requests)
                for request in requests:
                    if request.task_name not in self._logged_routing_tasks:
                        logging.info(
                            "routed model task: task=%s model=%s stream=%s track=%s frame=%s",
                            request.task_name,
                            request.model_name,
                            request.stream_id,
                            request.track_id,
                            request.frame_id,
                        )
                        self._logged_routing_tasks.add(request.task_name)
            self.json_writer.write(result)
            if hasattr(self.runtime_metrics, "observe"):
                self.runtime_metrics.observe(
                    result,
                    gpu_snapshot=self.gpu_monitor.snapshot() if hasattr(self.gpu_monitor, "snapshot") else None,
                )
        except Exception as exc:
            message = f"frame result handler failed: {exc}"
            logging.exception("frame result handler failed")
            self.pipeline_manager.set_error(message)
            self._stop_event.set()

    def handle_error(self, message: str) -> None:
        self.pipeline_manager.set_error(message)
        logging.error("pipeline error: %s", message)
        self.stop()
        raise RuntimeError(message)

    def on_helmet_event(self, event: HelmetEvent) -> None:
        if self.event_writer is not None and hasattr(self.event_writer, "write"):
            self.event_writer.write(event)
        logging.debug(
            "helmet violation: stream=%s track=%s frame=%s",
            event.stream_id,
            event.track_id,
            event.frame_id,
        )

    def on_vehicle_event(self, event: VehiclePassEvent) -> None:
        if self.event_writer is not None and hasattr(self.event_writer, "write"):
            self.event_writer.write(event)
        if hasattr(self.pipeline_manager, "register_plate_annotation"):
            self.pipeline_manager.register_plate_annotation(
                event.stream_id,
                event.track_id,
                {
                    "plate_text": event.plate_text,
                    "confidence": event.confidence,
                    "plate_bbox": {
                        "left": event.plate_bbox.left,
                        "top": event.plate_bbox.top,
                        "width": event.plate_bbox.width,
                        "height": event.plate_bbox.height,
                    },
                },
            )
        logging.debug(
            "vehicle pass: stream=%s track=%s plate=%s",
            event.stream_id,
            event.track_id,
            event.plate_text,
        )

    def on_pose_event(self, event: PoseEvent) -> None:
        if self.event_writer is not None and hasattr(self.event_writer, "write"):
            self.event_writer.write(event)

    def on_fire_smoke_event(self, event: FireSmokeEvent) -> None:
        if self.event_writer is not None and hasattr(self.event_writer, "write"):
            self.event_writer.write(event)
        logging.debug(
            "fire/smoke detection: stream=%s frame=%s status=%s",
            event.stream_id, event.frame_id, event.status,
        )

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
            "runtime_metrics": self.runtime_metrics.snapshot()
            if hasattr(self.runtime_metrics, "snapshot")
            else {},
            "controllers": {
                "fps": self.fps_controller.stats() if hasattr(self.fps_controller, "stats") else {},
                "backpressure": self.backpressure_controller.stats()
                if hasattr(self.backpressure_controller, "stats")
                else {},
            },
            "routing": {
                "policy": self.routing_policy.stats()
                if self.routing_policy is not None
                else {},
                "task_buffer": self.task_buffer.stats()
                if self.task_buffer is not None
                else {},
            },
            "helmet_worker": self.helmet_worker.stats()
            if self.helmet_worker is not None and hasattr(self.helmet_worker, "stats")
            else {},
            "plate_worker": self.plate_worker.stats()
            if self.plate_worker is not None and hasattr(self.plate_worker, "stats")
            else {},
            "events": self.event_writer.stats()
            if self.event_writer is not None and hasattr(self.event_writer, "stats")
            else {},
            "last_result": pipeline_state.get("last_result"),
            "is_running": pipeline_state.get("is_running"),
            "source_count": pipeline_state.get("source_count"),
            "last_error": pipeline_state.get("last_error"),
        }
