from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from threading import Event, Thread, current_thread
import time
from typing import Any

import cv2
import numpy as np

from app.application.helmet_service import TensorRTHelmetBackend, crop_person_roi, letterbox, map_roi_bbox
from app.application.routing_policy import TaskRequest
from app.application.task_metrics import TaskExecutionMetrics
from app.domain.entities import BoundingBox


@dataclass(frozen=True)
class PlateDetection:
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class PlateRecognition:
    stream_id: str
    vehicle_track_id: int
    frame_id: int
    plate_text: str
    confidence: float
    plate_bbox: BoundingBox


@dataclass(frozen=True)
class VehiclePassEvent:
    event_type: str
    stream_id: str
    track_id: int
    frame_id: int
    plate_text: str
    confidence: float
    plate_bbox: BoundingBox


class PlateTaskWorker:
    """Consume vehicle plate tasks and run detector→OCR as one task chain."""

    def __init__(self, task_buffer: Any, frame_store: Any, models: dict[str, Any] | None) -> None:
        self.task_buffer = task_buffer
        self.frame_store = frame_store
        self.models = models or {}
        self._handler = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._processor: PlateTaskProcessor | None = None
        self._error: str | None = None
        self._history: dict[tuple[str, int], list[PlateRecognition]] = {}
        self._metrics = TaskExecutionMetrics()

    def set_event_handler(self, handler) -> None:
        self._handler = handler

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="plate-task-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread() and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "initialized": self._processor is not None,
            "error": self._error,
            "tracked_vehicles": len(self._history),
            "processing_scope": "detector_ocr_pre_post",
            **self._metrics.stats(),
        }

    def _run(self) -> None:
        while not self._stop_event.wait(0.02):
            if not self.task_buffer.has_pending("plate_detector"):
                continue
            if self._processor is None:
                try:
                    detector = self.models.get("plate_detector")
                    ocr = self.models.get("plate_ocr")
                    if detector is None or ocr is None:
                        raise RuntimeError("plate detector and OCR models must both be configured")
                    if ocr.config_path is None:
                        raise RuntimeError("plate OCR model requires a config file")
                    if not Path(ocr.config_path).is_file():
                        raise RuntimeError(
                            f"plate OCR config file not found: {ocr.config_path}"
                        )
                    self._processor = PlateTaskProcessor(
                        TensorRTHelmetBackend(str(detector.engine_path)),
                        TensorRTHelmetBackend(str(ocr.engine_path)),
                        ocr_config_path=ocr.config_path,
                    )
                    logging.info("plate detector and OCR TensorRT backends initialized")
                except Exception as exc:  # pragma: no cover - target-device dependency
                    self._error = str(exc)
                    logging.error("plate worker initialization failed: %s", exc)
                    self.task_buffer.drain(task_name="plate_detector")
                    return
            for request in self.task_buffer.drain(2, task_name="plate_detector"):
                frame = self.frame_store.get_bgr(request.stream_id, request.frame_id, consumer="plate_detector")
                if frame is None:
                    self._metrics.missing_frames += 1
                    continue
                try:
                    started = time.monotonic()
                    self._metrics.record_queue_wait(
                        (started - request.submitted_at_monotonic) * 1000.0
                    )
                    processing_started = time.monotonic()
                    recognition = self._processor.process(request, frame)
                    self._metrics.record_inference((time.monotonic() - processing_started) * 1000.0)
                    self._metrics.processed += 1
                    self._metrics.record_task_latency(
                        (time.monotonic() - request.submitted_at_monotonic) * 1000.0
                    )
                    if recognition is None or not recognition.plate_text:
                        continue
                    key = (recognition.stream_id, recognition.vehicle_track_id)
                    values = self._history.setdefault(key, [])
                    values.append(recognition)
                    values[:] = values[-10:]
                    if len(values) >= 3:
                        best = max(values, key=lambda item: item.confidence)
                        if sum(item.plate_text == best.plate_text for item in values[-5:]) >= 3:
                            event = VehiclePassEvent(
                                "vehicle_pass",
                                best.stream_id,
                                best.vehicle_track_id,
                                best.frame_id,
                                best.plate_text,
                                best.confidence,
                                best.plate_bbox,
                            )
                            if self._handler is not None:
                                self._handler(event)
                            values.clear()
                except Exception as exc:
                    self._error = str(exc)
                    self._metrics.errors += 1
                    logging.exception("plate task failed")


def decode_plate_detector_output(
    output: np.ndarray,
    *,
    roi_shape: tuple[int, int],
    input_size: int = 384,
    confidence_threshold: float = 0.4,
) -> tuple[PlateDetection, ...]:
    """Decode FastALPR end2end output: [batch_index,x1,y1,x2,y2,class,score]."""
    raw = np.asarray(output).reshape(-1, 7)
    roi_height, roi_width = roi_shape[:2]
    ratio = min(input_size / roi_width, input_size / roi_height)
    pad_x = (input_size - roi_width * ratio) / 2.0
    pad_y = (input_size - roi_height * ratio) / 2.0
    results: list[PlateDetection] = []
    for row in raw:
        confidence = float(row[6])
        if confidence < confidence_threshold:
            continue
        left = max(min((float(row[1]) - pad_x) / ratio, roi_width), 0.0)
        top = max(min((float(row[2]) - pad_y) / ratio, roi_height), 0.0)
        right = max(min((float(row[3]) - pad_x) / ratio, roi_width), 0.0)
        bottom = max(min((float(row[4]) - pad_y) / ratio, roi_height), 0.0)
        if right > left and bottom > top:
            results.append(PlateDetection(confidence, BoundingBox(left, top, right - left, bottom - top)))
    return tuple(results)


def decode_ocr_output(output: np.ndarray, *, alphabet: str, pad_char: str = "_") -> tuple[str, float]:
    raw = np.asarray(output)
    if raw.ndim == 2:
        raw = raw.reshape(1, *raw.shape)
    indices = np.argmax(raw, axis=-1)[0]
    probabilities = np.max(raw, axis=-1)[0]
    chars = [alphabet[int(index)] for index in indices if int(index) < len(alphabet)]
    text = "".join(chars).rstrip(pad_char)
    confidence = float(np.mean(probabilities[: len(chars)])) if chars else 0.0
    return text, confidence


class PlateTaskProcessor:
    def __init__(self, detector_backend: Any, ocr_backend: Any, *, ocr_config_path: Path) -> None:
        self.detector_backend = detector_backend
        self.ocr_backend = ocr_backend
        self.ocr_config = self._load_ocr_config(ocr_config_path)

    def process(self, request: TaskRequest, frame: np.ndarray) -> PlateRecognition | None:
        vehicle_roi, vehicle_rect = crop_person_roi(frame, request.bbox, padding_ratio=0.02)
        detector_input, _, _, _ = letterbox(vehicle_roi, 384, 384)
        detector_tensor = np.transpose(
            cv2.cvtColor(detector_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            (2, 0, 1),
        )[None, ...]
        plate_detections = decode_plate_detector_output(
            self.detector_backend.infer(detector_tensor),
            roi_shape=vehicle_roi.shape[:2],
        )
        if not plate_detections:
            return None
        plate = max(plate_detections, key=lambda item: item.confidence)
        full_plate_bbox = map_roi_bbox(plate.bbox, vehicle_rect)
        crop = frame[
            max(int(full_plate_bbox.top), 0) : int(full_plate_bbox.top + full_plate_bbox.height),
            max(int(full_plate_bbox.left), 0) : int(full_plate_bbox.left + full_plate_bbox.width),
        ]
        if crop.size == 0:
            return None
        ocr_image = cv2.resize(crop, (self.ocr_config["img_width"], self.ocr_config["img_height"]))
        if self.ocr_config.get("image_color_mode", "rgb") == "rgb":
            ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB)
        else:
            ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
        ocr_output = self.ocr_backend.infer(ocr_image[None, ...].astype(np.uint8))
        text, confidence = decode_ocr_output(
            ocr_output,
            alphabet=self.ocr_config["alphabet"],
            pad_char=self.ocr_config.get("pad_char", "_"),
        )
        return PlateRecognition(
            request.stream_id,
            request.track_id,
            request.frame_id,
            text,
            confidence * plate.confidence,
            full_plate_bbox,
        )

    @staticmethod
    def _load_ocr_config(path: Path) -> dict[str, Any]:
        import yaml

        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}
