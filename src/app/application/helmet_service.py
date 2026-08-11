from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import ctypes
import ctypes.util
from threading import Event, Thread, current_thread
import time
from typing import Any, Protocol

import cv2
import numpy as np

from app.domain.entities import BoundingBox
from app.application.routing_policy import TaskRequest


@dataclass(frozen=True)
class HelmetDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class HelmetAssessment:
    stream_id: str
    track_id: int
    frame_id: int
    status: str
    confidence: float
    detection: HelmetDetection | None = None


@dataclass(frozen=True)
class HelmetEvent:
    event_type: str
    stream_id: str
    track_id: int
    frame_id: int
    status: str
    confidence: float
    bbox: BoundingBox
    timestamp: datetime


class HelmetBackend(Protocol):
    def infer(self, image: np.ndarray) -> np.ndarray:
        """Return raw YOLOv8 output shaped like [1, 4 + classes, anchors]."""


class _CtypesCudaRuntime:
    """Minimal libcudart bridge used when cuda-python is not installed."""

    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    def __init__(self) -> None:
        candidates = [
            ctypes.util.find_library("cudart"),
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so",
        ]
        library_path = next((path for path in candidates if path), None)
        if library_path is None:
            raise RuntimeError("libcudart.so was not found")
        self._library = ctypes.CDLL(library_path)
        self._library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._library.cudaMalloc.restype = ctypes.c_int
        self._library.cudaFree.argtypes = [ctypes.c_void_p]
        self._library.cudaFree.restype = ctypes.c_int
        self._library.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._library.cudaMemcpy.restype = ctypes.c_int
        self._library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._library.cudaMemcpyAsync.restype = ctypes.c_int
        self._library.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._library.cudaStreamCreate.restype = ctypes.c_int
        self._library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._library.cudaStreamSynchronize.restype = ctypes.c_int

    def cudaMalloc(self, size: int):
        pointer = ctypes.c_void_p()
        error = self._library.cudaMalloc(ctypes.byref(pointer), size)
        return error, pointer.value or 0

    def cudaFree(self, pointer: int):
        return (self._library.cudaFree(ctypes.c_void_p(pointer)),)

    def cudaMemcpy(self, destination: int, source: int, size: int, kind: int):
        return (
            self._library.cudaMemcpy(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                size,
                kind,
            ),
        )

    def cudaMemcpyAsync(self, destination: int, source: int, size: int, kind: int, stream: int):
        return (
            self._library.cudaMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                size,
                kind,
                ctypes.c_void_p(stream),
            ),
        )

    def cudaStreamCreate(self):
        stream = ctypes.c_void_p()
        error = self._library.cudaStreamCreate(ctypes.byref(stream))
        return error, stream.value or 0

    def cudaStreamSynchronize(self, stream: int):
        return (self._library.cudaStreamSynchronize(ctypes.c_void_p(stream)),)


try:  # TensorRT requires output allocators to derive from its Python interface.
    import tensorrt as _trt_for_allocator

    _TensorRTOutputAllocatorBase = _trt_for_allocator.IOutputAllocator
except ImportError:  # pragma: no cover - TensorRT is only available on target
    class _TensorRTOutputAllocatorBase:
        pass


class _TensorRTOutputAllocator(_TensorRTOutputAllocatorBase):
    """CUDA device-buffer allocator for TensorRT dynamic output tensors."""

    def __init__(self, cudart: Any) -> None:
        if hasattr(super(), "__init__"):
            super().__init__()
        self._cudart = cudart
        self.pointer = 0
        self.size = 0
        self.shape: tuple[int, ...] | None = None
        self.error: str | None = None

    def reallocate_output(self, _name: str, _memory: int, size: int, _alignment: int) -> int:
        if self.pointer:
            self._cudart.cudaFree(self.pointer)
            self.pointer = 0
        requested_size = max(int(size), 1)
        error, pointer = self._cudart.cudaMalloc(requested_size)
        if int(error) != 0 or not pointer:
            self.error = (
                f"cudaMalloc(dynamic output) failed: requested={requested_size} "
                f"error={error} pointer={pointer!r}"
            )
            return 0
        self.pointer = int(pointer)
        self.size = requested_size
        return self.pointer

    def notify_shape(self, _name: str, shape: Any) -> None:
        self.shape = tuple(int(value) for value in shape)


class TensorRTHelmetBackend:
    """TensorRT engine backend using the CUDA Python bindings on Jetson."""

    def __init__(self, engine_path: str) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - target-device dependency
            raise RuntimeError("TensorRT helmet inference requires `tensorrt`") from exc
        try:
            from cuda import cudart
            logging.info("helmet backend using cuda-python")
        except ImportError:
            cudart = _CtypesCudaRuntime()
            logging.info("helmet backend using libcudart ctypes fallback")
        self._trt = trt
        self._cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as stream:
            engine_bytes = stream.read()
        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        self._input_name = next(
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
        )
        self._output_names = tuple(
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        )
        error, stream_handle = cudart.cudaStreamCreate()
        self._check(error, "cudaStreamCreate")
        self._stream = stream_handle

    def infer(self, image: np.ndarray) -> np.ndarray:
        tensor = np.ascontiguousarray(image)
        self._context.set_input_shape(self._input_name, tensor.shape)
        # Dynamic TensorRT engines can keep output dimensions unresolved until
        # shape inference is explicitly requested after setting the input.
        infer_shapes = getattr(self._context, "infer_shapes", None)
        if infer_shapes is not None:
            unresolved = infer_shapes()
            if unresolved:
                raise RuntimeError(
                    f"TensorRT could not infer shapes for tensors: {unresolved}"
                )
        allocations: list[tuple[int, np.ndarray]] = []
        static_allocations: dict[str, tuple[int, np.ndarray]] = {}
        dynamic_allocators: dict[str, _TensorRTOutputAllocator] = {}
        try:
            error, input_device = self._cudart.cudaMalloc(tensor.nbytes)
            self._check(error, "cudaMalloc(input)")
            allocations.append((input_device, tensor))
            self._context.set_tensor_address(self._input_name, int(input_device))
            error = self._cudart.cudaMemcpyAsync(
                input_device,
                tensor.ctypes.data,
                tensor.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )[0]
            self._check(error, "cudaMemcpyAsync(input)")

            outputs: list[np.ndarray] = []
            for output_name in self._output_names:
                shape = tuple(int(value) for value in self._context.get_tensor_shape(output_name))
                if any(value <= 0 for value in shape):
                    allocator = _TensorRTOutputAllocator(self._cudart)
                    self._context.set_output_allocator(output_name, allocator)
                    dynamic_allocators[output_name] = allocator
                    outputs.append(None)
                    continue
                output_dtype = self._trt.nptype(self._engine.get_tensor_dtype(output_name))
                output = np.empty(shape, dtype=output_dtype)
                error, output_device = self._cudart.cudaMalloc(output.nbytes)
                self._check(error, f"cudaMalloc({output_name})")
                allocations.append((output_device, output))
                self._context.set_tensor_address(output_name, int(output_device))
                outputs.append(output)
                static_allocations[output_name] = (output_device, output)

            if not self._context.execute_async_v3(self._stream):
                raise RuntimeError("TensorRT helmet execution failed")
            error = self._cudart.cudaStreamSynchronize(self._stream)[0]
            self._check(error, "cudaStreamSynchronize")
            for output_index, output_name in enumerate(self._output_names):
                allocator = dynamic_allocators.get(output_name)
                if allocator is not None:
                    if allocator.error:
                        raise RuntimeError(allocator.error)
                    if not allocator.pointer or not allocator.shape:
                        raise RuntimeError(
                            f"TensorRT did not provide dynamic output shape for {output_name}"
                        )
                    output_dtype = self._trt.nptype(self._engine.get_tensor_dtype(output_name))
                    output = np.empty(allocator.shape, dtype=output_dtype)
                    error = self._cudart.cudaMemcpy(
                        output.ctypes.data,
                        allocator.pointer,
                        output.nbytes,
                        self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )[0]
                    self._check(error, f"cudaMemcpy({output_name})")
                    outputs[output_index] = output
                    continue
                device, output = static_allocations[output_name]
                error = self._cudart.cudaMemcpy(
                    output.ctypes.data,
                    device,
                    output.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )[0]
                self._check(error, f"cudaMemcpy({output_name})")
            if any(output is None for output in outputs):
                raise RuntimeError("TensorRT returned an empty output buffer")
            return outputs[0]
        finally:
            for device, _ in allocations:
                self._cudart.cudaFree(device)
            for allocator in dynamic_allocators.values():
                if allocator.pointer:
                    self._cudart.cudaFree(allocator.pointer)

    @staticmethod
    def _check(error: Any, operation: str) -> None:
        if int(error) != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {error}")


def load_labels(path: str) -> tuple[str, ...]:
    with open(path, "r", encoding="utf-8") as stream:
        return tuple(line.strip() for line in stream if line.strip())


def crop_person_roi(
    frame: np.ndarray,
    bbox: BoundingBox,
    *,
    padding_ratio: float = 0.05,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a person ROI and return it with its original-frame rectangle."""
    if frame is None or frame.ndim < 2:
        raise ValueError("frame must be a non-empty image")
    height, width = frame.shape[:2]
    pad_x = max(int(bbox.width * padding_ratio), 0)
    pad_y = max(int(bbox.height * padding_ratio), 0)
    left = max(int(round(bbox.left)) - pad_x, 0)
    top = max(int(round(bbox.top)) - pad_y, 0)
    right = min(int(round(bbox.left + bbox.width)) + pad_x, width)
    bottom = min(int(round(bbox.top + bbox.height)) + pad_y, height)
    if right <= left or bottom <= top:
        raise ValueError("person bbox does not intersect the frame")
    return frame[top:bottom, left:right].copy(), (left, top, right - left, bottom - top)


def map_roi_bbox(bbox: BoundingBox, roi_rect: tuple[int, int, int, int]) -> BoundingBox:
    left, top, _, _ = roi_rect
    return BoundingBox(
        left=bbox.left + left,
        top=bbox.top + top,
        width=bbox.width,
        height=bbox.height,
    )


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, float, float]:
    source_height, source_width = image.shape[:2]
    ratio = min(width / source_width, height / source_height)
    resized_width = max(int(round(source_width * ratio)), 1)
    resized_height = max(int(round(source_height * ratio)), 1)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (width - resized_width) / 2.0
    pad_y = (height - resized_height) / 2.0
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, ratio, pad_x, pad_y


def decode_yolov8_output(
    output: np.ndarray,
    *,
    roi_shape: tuple[int, int],
    input_width: int = 640,
    input_height: int = 640,
    labels: tuple[str, ...] = (),
    confidence_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
) -> tuple[HelmetDetection, ...]:
    """Decode an Ultralytics YOLOv8 detect head without an NMS plugin."""
    raw = np.asarray(output)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2:
        raise ValueError(f"YOLOv8 output must have two dimensions after squeeze, got {raw.shape}")
    if raw.shape[0] <= 128 and (raw.shape[1] > raw.shape[0] or raw.shape[1] < 5):
        raw = raw.T
    if raw.shape[1] < 5:
        raise ValueError(f"YOLOv8 output has too few columns: {raw.shape}")

    roi_height, roi_width = roi_shape[:2]
    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    for row in raw:
        class_id = int(np.argmax(row[4:]))
        confidence = float(row[4 + class_id])
        if confidence < confidence_threshold:
            continue
        center_x, center_y, box_width, box_height = map(float, row[:4])
        left = (center_x - box_width / 2.0 - 0.0) / 1.0
        top = (center_y - box_height / 2.0 - 0.0) / 1.0
        # The caller supplies model-space boxes.  Undo letterbox below.
        boxes.append([left, top, box_width, box_height])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return ()

    # Recover the letterbox transform used for the ROI.
    ratio = min(input_width / roi_width, input_height / roi_height)
    pad_x = (input_width - roi_width * ratio) / 2.0
    pad_y = (input_height - roi_height * ratio) / 2.0
    decoded: list[HelmetDetection] = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        left = max(min((box[0] - pad_x) / ratio, roi_width), 0.0)
        top = max(min((box[1] - pad_y) / ratio, roi_height), 0.0)
        right = max(min((box[0] + box[2] - pad_x) / ratio, roi_width), 0.0)
        bottom = max(min((box[1] + box[3] - pad_y) / ratio, roi_height), 0.0)
        if right <= left or bottom <= top:
            continue
        decoded.append(
            HelmetDetection(
                class_id=class_id,
                class_name=labels[class_id] if class_id < len(labels) else str(class_id),
                confidence=score,
                bbox=BoundingBox(left, top, right - left, bottom - top),
            )
        )

    kept: list[HelmetDetection] = []
    for class_id in sorted({item.class_id for item in decoded}):
        candidates = [item for item in decoded if item.class_id == class_id]
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        while candidates:
            selected = candidates.pop(0)
            kept.append(selected)
            candidates = [
                item
                for item in candidates
                if bbox_iou(selected.bbox, item.bbox) < nms_iou_threshold
            ]
    return tuple(kept)


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.left, right.left)
    y1 = max(left.top, right.top)
    x2 = min(left.left + left.width, right.left + right.width)
    y2 = min(left.top + left.height, right.top + right.height)
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_left = max(left.width, 0.0) * max(left.height, 0.0)
    area_right = max(right.width, 0.0) * max(right.height, 0.0)
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def bbox_iom(inner: BoundingBox, container: BoundingBox) -> float:
    x1 = max(inner.left, container.left)
    y1 = max(inner.top, container.top)
    x2 = min(inner.left + inner.width, container.left + container.width)
    y2 = min(inner.top + inner.height, container.top + container.height)
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area = max(inner.width, 0.0) * max(inner.height, 0.0)
    return intersection / area if area > 0 else 0.0


class HelmetAssociator:
    def __init__(self, *, head_ratio: float = 0.45, min_iom: float = 0.25) -> None:
        self.head_ratio = head_ratio
        self.min_iom = min_iom

    def associate(
        self,
        person_bbox: BoundingBox,
        detections: tuple[HelmetDetection, ...],
        *,
        stream_id: str,
        track_id: int,
        frame_id: int,
    ) -> HelmetAssessment:
        head = BoundingBox(
            person_bbox.left,
            person_bbox.top,
            person_bbox.width,
            person_bbox.height * self.head_ratio,
        )
        candidates: list[tuple[float, HelmetDetection]] = []
        for detection in detections:
            name = detection.class_name.lower().replace("_", "-")
            if name not in {"hardhat", "no-hardhat", "helmet", "no-helmet"}:
                continue
            center_x = detection.bbox.left + detection.bbox.width / 2.0
            center_y = detection.bbox.top + detection.bbox.height / 2.0
            if not (head.left <= center_x <= head.left + head.width and head.top <= center_y <= head.top + head.height):
                continue
            iom = bbox_iom(detection.bbox, head)
            if iom < self.min_iom:
                continue
            candidates.append((detection.confidence * 0.7 + iom * 0.3, detection))
        if not candidates:
            return HelmetAssessment(stream_id, track_id, frame_id, "unknown", 0.0)
        _, selected = max(candidates, key=lambda item: item[0])
        name = selected.class_name.lower().replace("_", "-")
        status = "not_wearing" if name in {"no-hardhat", "no-helmet"} else "wearing"
        return HelmetAssessment(stream_id, track_id, frame_id, status, selected.confidence, selected)


class HelmetTaskProcessor:
    """Run one helmet task after a frame provider supplies the source image."""

    def __init__(self, backend: HelmetBackend, labels: tuple[str, ...], *, input_size=(640, 640)) -> None:
        self.backend = backend
        self.labels = labels
        self.input_width, self.input_height = input_size
        self.associator = HelmetAssociator()

    def process(self, request: TaskRequest, frame: np.ndarray) -> HelmetAssessment:
        return self.process_batch(((request, frame),))[0]

    def process_batch(
        self, tasks: tuple[tuple[TaskRequest, np.ndarray], ...]
    ) -> tuple[HelmetAssessment, ...]:
        """Run one TensorRT request for a non-empty batch of person ROIs."""
        if not tasks:
            return ()
        prepared: list[tuple[TaskRequest, tuple[int, int], tuple[int, int, int, int], np.ndarray]] = []
        for request, frame in tasks:
            roi, roi_rect = crop_person_roi(frame, request.bbox)
            model_image, _, _, _ = letterbox(roi, self.input_width, self.input_height)
            tensor = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            prepared.append((request, roi.shape[:2], roi_rect, np.transpose(tensor, (2, 0, 1))))

        output = np.asarray(self.backend.infer(np.stack([item[3] for item in prepared], axis=0)))
        if output.ndim < 3 or output.shape[0] != len(prepared):
            raise RuntimeError(
                f"PPE engine output batch {output.shape} does not match input batch {len(prepared)}"
            )

        assessments: list[HelmetAssessment] = []
        for index, (request, roi_shape, roi_rect, _) in enumerate(prepared):
            detections = decode_yolov8_output(
                output[index : index + 1],
                roi_shape=roi_shape,
                input_width=self.input_width,
                input_height=self.input_height,
                labels=self.labels,
            )
            mapped = tuple(
                HelmetDetection(
                    item.class_id,
                    item.class_name,
                    item.confidence,
                    map_roi_bbox(item.bbox, roi_rect),
                )
                for item in detections
            )
            assessments.append(
                self.associator.associate(
                    request.bbox,
                    mapped,
                    stream_id=request.stream_id,
                    track_id=request.track_id,
                    frame_id=request.frame_id,
                )
            )
        return tuple(assessments)


@dataclass
class _HelmetTrackState:
    last_frame_id: int = -1
    not_wearing_frames: int = 0
    wearing_frames: int = 0
    confirmed_status: str = "unknown"
    last_event_frame: int = -1


class HelmetEventTracker:
    """Turn frame-level helmet assessments into stable violation events."""

    def __init__(self, *, confirm_frames: int = 5, cooldown_frames: int = 30) -> None:
        self.confirm_frames = max(int(confirm_frames), 1)
        self.cooldown_frames = max(int(cooldown_frames), 0)
        self._states: dict[tuple[str, int], _HelmetTrackState] = {}

    def update(self, assessment: HelmetAssessment, person_bbox: BoundingBox) -> HelmetEvent | None:
        key = (assessment.stream_id, assessment.track_id)
        state = self._states.setdefault(key, _HelmetTrackState())
        if state.last_frame_id >= 0 and assessment.frame_id <= state.last_frame_id:
            return None
        state.last_frame_id = assessment.frame_id
        if assessment.status == "not_wearing":
            state.not_wearing_frames += 1
            state.wearing_frames = 0
        elif assessment.status == "wearing":
            state.wearing_frames += 1
            state.not_wearing_frames = 0
        else:
            return None
        if assessment.status == "not_wearing":
            if state.not_wearing_frames < self.confirm_frames:
                return None
            if state.confirmed_status == "not_wearing":
                return None
            if state.last_event_frame >= 0 and assessment.frame_id - state.last_event_frame < self.cooldown_frames:
                return None
            state.confirmed_status = "not_wearing"
            state.last_event_frame = assessment.frame_id
            return HelmetEvent(
                event_type="helmet_violation",
                stream_id=assessment.stream_id,
                track_id=assessment.track_id,
                frame_id=assessment.frame_id,
                status=assessment.status,
                confidence=assessment.confidence,
                bbox=person_bbox,
                timestamp=datetime.now(timezone.utc),
            )
        if assessment.status == "wearing" and state.wearing_frames >= self.confirm_frames:
            state.confirmed_status = "wearing"
        return None


class HelmetTaskExecutor:
    """Consume routed helmet tasks without blocking the DeepStream probe."""

    def __init__(self, processor: HelmetTaskProcessor, event_tracker: HelmetEventTracker) -> None:
        self.processor = processor
        self.event_tracker = event_tracker
        self.processed = 0
        self.missing_frames = 0
        self.errors = 0
        self.batch_sizes: list[int] = []
        self.queue_wait_ms: list[float] = []
        self.inference_ms: list[float] = []
        self.task_latency_ms: list[float] = []

    def process_requests(
        self, requests: tuple[TaskRequest, ...], frame_store: Any
    ) -> tuple[HelmetEvent, ...]:
        events: list[HelmetEvent] = []
        tasks: list[tuple[TaskRequest, np.ndarray]] = []
        for request in requests:
            frame = frame_store.get_bgr(request.stream_id, request.frame_id)
            if frame is None:
                self.missing_frames += 1
                continue
            tasks.append((request, frame))
        if not tasks:
            return ()
        inference_started = time.monotonic()
        self.queue_wait_ms.extend(
            max((inference_started - request.submitted_at_monotonic) * 1000.0, 0.0)
            for request, _ in tasks
        )
        try:
            assessments = self.processor.process_batch(tuple(tasks))
        except Exception:
            self.errors += len(tasks)
            logging.exception("helmet micro-batch failed: size=%s", len(tasks))
            return ()
        self.batch_sizes.append(len(tasks))
        now = time.monotonic()
        self.inference_ms.append((now - inference_started) * 1000.0)
        self.task_latency_ms.extend(
            max((now - request.submitted_at_monotonic) * 1000.0, 0.0)
            for request, _ in tasks
        )
        for (request, _), assessment in zip(tasks, assessments):
            self.processed += 1
            event = self.event_tracker.update(assessment, request.bbox)
            if event is not None:
                events.append(event)
        return tuple(events)

    def stats(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "missing_frames": self.missing_frames,
            "errors": self.errors,
            "batches": len(self.batch_sizes),
            "batch_size": _sample_summary(self.batch_sizes),
            "queue_wait_ms": _sample_summary(self.queue_wait_ms),
            "inference_ms": _sample_summary(self.inference_ms),
            "task_latency_ms": _sample_summary(self.task_latency_ms),
        }


class HelmetTaskWorker:
    """Lazy, dedicated worker for the configured PPE TensorRT engine."""

    def __init__(
        self, task_buffer: Any, frame_store: Any, model_settings: Any | None, task_settings: Any | None = None
    ) -> None:
        self.task_buffer = task_buffer
        self.frame_store = frame_store
        self.model_settings = model_settings
        self.task_settings = task_settings
        self._event_handler = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._executor: HelmetTaskExecutor | None = None
        self._init_error: str | None = None
        self._logged_processing = False
        self._disabled = False

    def set_event_handler(self, handler) -> None:
        self._event_handler = handler

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="helmet-task-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread() and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def stats(self) -> dict[str, Any]:
        executor = self._executor
        executor_stats = executor.stats() if executor is not None else {}
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "initialized": executor is not None,
            "disabled": self._disabled,
            "init_error": self._init_error,
            "micro_batch_size": self._micro_batch_size(),
            "micro_batch_wait_ms": self._micro_batch_wait_ms(),
            **executor_stats,
        }

    def _run(self) -> None:
        while not self._stop_event.wait(0.02):
            if self.model_settings is None:
                continue
            if self._disabled:
                return
            if not self.task_buffer.has_pending("helmet"):
                continue
            if self._executor is None:
                try:
                    labels_path = self.model_settings.labels_path
                    if labels_path is None:
                        raise RuntimeError("helmet model requires a labels file")
                    labels = load_labels(str(labels_path))
                    backend = TensorRTHelmetBackend(str(self.model_settings.engine_path))
                    self._executor = HelmetTaskExecutor(
                        HelmetTaskProcessor(
                            backend,
                            labels,
                            input_size=(self.model_settings.input_width, self.model_settings.input_height),
                        ),
                        HelmetEventTracker(),
                    )
                    logging.info("helmet TensorRT backend initialized: %s", self.model_settings.engine_path)
                except Exception as exc:  # pragma: no cover - target-device dependency
                    self._init_error = str(exc)
                    logging.error("helmet worker initialization failed: %s", exc)
                    self.task_buffer.drain(task_name="helmet")
                    self._disabled = True
                    return
            requests = self._collect_requests()
            events = self._executor.process_requests(requests, self.frame_store)
            if self._executor.processed > 0 and not self._logged_processing:
                logging.info("helmet task processed successfully: count=%s", self._executor.processed)
                self._logged_processing = True
            if self._event_handler is not None:
                for event in events:
                    self._event_handler(event)

    def _micro_batch_size(self) -> int:
        return max(int(getattr(self.task_settings, "micro_batch_size", 1)), 1)

    def _micro_batch_wait_ms(self) -> int:
        return max(int(getattr(self.task_settings, "micro_batch_wait_ms", 0)), 0)

    def _collect_requests(self) -> tuple[TaskRequest, ...]:
        maximum = self._micro_batch_size()
        requests = list(self.task_buffer.drain(maximum, task_name="helmet"))
        if len(requests) >= maximum:
            return tuple(requests)
        deadline = time.monotonic() + self._micro_batch_wait_ms() / 1000.0
        while len(requests) < maximum and not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._stop_event.wait(min(remaining, 0.005))
            requests.extend(self.task_buffer.drain(maximum - len(requests), task_name="helmet"))
        return tuple(requests)


def _sample_summary(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "average": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "samples": len(ordered),
        "average": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "max": round(ordered[-1], 3),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
