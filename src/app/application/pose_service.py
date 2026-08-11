from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Event, Thread, current_thread
import time
from typing import Any

import numpy as np

from app.application.helmet_service import TensorRTHelmetBackend, crop_person_roi, letterbox
from app.application.routing_policy import TaskRequest
from app.application.task_metrics import TaskExecutionMetrics


@dataclass(frozen=True)
class PoseEvent:
    event_type: str
    stream_id: str
    track_id: int
    frame_id: int
    keypoints: tuple[tuple[float, float, float], ...]
    confidence: float


def decode_pose_output(output: np.ndarray, roi_shape: tuple[int, int], input_size: int = 640) -> tuple[tuple[float, float, float], ...] | None:
    raw = np.asarray(output)
    if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
        raw = raw.transpose(0, 2, 1)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] < 56:
        return None
    row = raw[int(np.argmax(raw[:, 4]))]
    confidence = float(row[4])
    if confidence < 0.10:
        return None
    height, width = roi_shape[:2]
    ratio = min(input_size / width, input_size / height)
    pad_x = (input_size - width * ratio) / 2.0
    pad_y = (input_size - height * ratio) / 2.0
    points = []
    for index in range(17):
        x = (float(row[5 + index * 3]) - pad_x) / ratio
        y = (float(row[6 + index * 3]) - pad_y) / ratio
        score = float(row[7 + index * 3])
        points.append((max(0.0, min(x, width)), max(0.0, min(y, height)), score))
    return tuple(points)


class PoseTaskWorker:
    def __init__(self, task_buffer: Any, frame_store: Any, model_settings: Any | None) -> None:
        self.task_buffer = task_buffer
        self.frame_store = frame_store
        self.model_settings = model_settings
        self._handler = None
        self._backend = None
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._error: str | None = None
        self._processed = 0
        self._emitted = 0
        self._metrics = TaskExecutionMetrics()
        self._logged_output = False

    def set_event_handler(self, handler) -> None:
        self._handler = handler

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="pose-task-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread is not current_thread() and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "initialized": self._backend is not None,
            "error": self._error,
            "emitted": self._emitted,
            **self._metrics.stats(),
        }

    def _run(self) -> None:
        while not self._stop_event.wait(0.02):
            requests = self.task_buffer.drain(1, task_name="pose")
            if not requests:
                continue
            if self._backend is None:
                try:
                    if self.model_settings is None:
                        raise RuntimeError("pose model is not configured")
                    self._backend = TensorRTHelmetBackend(str(self.model_settings.engine_path))
                    logging.info("pose TensorRT backend initialized: %s", self.model_settings.engine_path)
                except Exception as exc:
                    self._error = str(exc)
                    logging.error("pose worker initialization failed: %s", exc)
                    continue
            for request in requests:
                frame = self.frame_store.get_bgr(request.stream_id, request.frame_id, consumer="pose")
                if frame is None:
                    self._metrics.missing_frames += 1
                    continue
                try:
                    started = time.monotonic()
                    self._metrics.record_queue_wait(
                        (started - request.submitted_at_monotonic) * 1000.0
                    )
                    roi, _ = crop_person_roi(frame, request.bbox)
                    image, _, _, _ = letterbox(roi, 640, 640)
                    tensor = np.transpose(image[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
                    inference_started = time.monotonic()
                    output = self._backend.infer(tensor)
                    self._metrics.record_inference((time.monotonic() - inference_started) * 1000.0)
                    if not self._logged_output:
                        raw = np.asarray(output)
                        logging.info("pose output shape=%s max=%.4f", raw.shape, float(np.max(raw)))
                        self._logged_output = True
                    keypoints = decode_pose_output(output, roi.shape[:2])
                    self._metrics.processed += 1
                    self._processed = self._metrics.processed
                    self._metrics.record_task_latency(
                        (time.monotonic() - request.submitted_at_monotonic) * 1000.0
                    )
                    if keypoints is not None and self._handler is not None:
                        self._handler(PoseEvent("pose_observation", request.stream_id, request.track_id, request.frame_id, keypoints, float(request.confidence)))
                        self._emitted += 1
                except Exception as exc:
                    self._error = str(exc)
                    self._metrics.errors += 1
                    logging.exception("pose task failed")
