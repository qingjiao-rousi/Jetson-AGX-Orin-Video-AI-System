from __future__ import annotations

"""火焰/烟雾整帧检测的 batch=1 异步 worker 与后处理。"""

from dataclasses import dataclass
import logging
from threading import Event, Thread, current_thread
import time
from typing import Any

import cv2
import numpy as np

# 复用通用 TensorRT backend；烟火任务直接处理整帧而不是 person ROI。
from app.application.helmet_service import TensorRTHelmetBackend, letterbox
from app.application.routing_policy import TaskRequest
from app.application.task_metrics import TaskExecutionMetrics


@dataclass(frozen=True)
class FireSmokeEvent:
    """整帧 fire/smoke 模型的一条检测事件，bbox 使用源帧坐标。"""
    event_type: str
    stream_id: str
    frame_id: int
    status: str
    confidence: float
    bbox: dict[str, float]


def decode_fire_smoke(
    output: np.ndarray,
    labels: tuple[str, ...],
    threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> tuple[tuple[str, float, dict[str, float]], ...]:
    """解码 YOLO 风格输出并按类别执行 OpenCV NMS，输出仍处于模型输入坐标系。"""
    raw = np.asarray(output)
    if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
        raw = raw.transpose(0, 2, 1)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] < 5:
        return ()
    candidates: dict[int, list[tuple[float, dict[str, float]]]] = {}
    for row in raw:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < threshold or class_id >= len(labels):
            continue
        candidates.setdefault(class_id, []).append((confidence, {"left": float(row[0]), "top": float(row[1]), "width": float(row[2]), "height": float(row[3])}))
    results = []
    # 每类各自 NMS，fire 与 smoke 的重叠框不互相抑制。
    for class_id, values in candidates.items():
        boxes = [[box["left"], box["top"], box["width"], box["height"]] for _, box in values]
        scores = [score for score, _ in values]
        keep = cv2.dnn.NMSBoxes(boxes, scores, threshold, iou_threshold)
        for index in np.asarray(keep).reshape(-1).tolist() if len(keep) else []:
            score, box = values[int(index)]
            results.append((labels[class_id], score, box))
    return tuple(results)


class FireSmokeTaskWorker:
    """独立消费 fire_smoke 帧触发任务的 batch=1 worker。

    与 PPE/pose 的目标 ROI 不同，路由层为它提交整帧虚拟 ROI；因此直接使用
    FrameStore 的整帧 BGR 图像。当前不实施微批，保持配置和 engine 的 batch 契约。
    """
    def __init__(self, task_buffer: Any, frame_store: Any, model_settings: Any | None) -> None:
        self.task_buffer = task_buffer
        self.frame_store = frame_store
        self.model_settings = model_settings
        self._handler = None
        self._backend = None
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._error: str | None = None
        self._labels = ("fire", "smoke")
        self._metrics = TaskExecutionMetrics()

    def set_event_handler(self, handler) -> None:
        self._handler = handler

    def start(self) -> None:
        """启动烟火 worker，真正加载 engine 延迟到首个任务。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="fire-smoke-task-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止轮询并有界等待线程退出。"""
        self._stop_event.set()
        if self._thread is not None and self._thread is not current_thread() and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "initialized": self._backend is not None,
            "error": self._error,
            **self._metrics.stats(),
        }

    def _run(self) -> None:
        """执行取整帧、letterbox、推理、反变换和逐检测事件回调。"""
        while not self._stop_event.wait(0.02):
            # 当前模型按单帧执行；独立任务队列仍可隔离它对其它任务的影响。
            requests = self.task_buffer.drain(1, task_name="fire_smoke")
            if not requests:
                continue
            if self._backend is None:
                try:
                    if self.model_settings is None:
                        raise RuntimeError("fire/smoke model is not configured")
                    self._backend = TensorRTHelmetBackend(str(self.model_settings.engine_path))
                    if self.model_settings.labels_path and self.model_settings.labels_path.is_file():
                        self._labels = tuple(line.strip() for line in self.model_settings.labels_path.read_text(encoding="utf-8").splitlines() if line.strip())
                    logging.info("fire/smoke TensorRT backend initialized: %s", self.model_settings.engine_path)
                except Exception as exc:
                    self._error = str(exc)
                    logging.error("fire/smoke worker initialization failed: %s", exc)
                    continue
            for request in requests:
                # 路由提交的是 frame trigger，因此 FrameStore 中的整帧就是模型输入来源。
                frame = self.frame_store.get_bgr(request.stream_id, request.frame_id, consumer="fire_smoke")
                if frame is None:
                    self._metrics.missing_frames += 1
                    continue
                try:
                    started = time.monotonic()
                    self._metrics.record_queue_wait(
                        (started - request.submitted_at_monotonic) * 1000.0
                    )
                    image, ratio, pad_x, pad_y = letterbox(frame, 640, 640)
                    tensor = np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
                    inference_started = time.monotonic()
                    output = self._backend.infer(tensor)
                    self._metrics.record_inference((time.monotonic() - inference_started) * 1000.0)
                    # decoder 给出模型输入坐标，必须撤销 letterbox 后再写入业务事件。
                    for status, confidence, box in decode_fire_smoke(output, self._labels):
                        box["left"] = max((box["left"] - pad_x) / ratio, 0.0)
                        box["top"] = max((box["top"] - pad_y) / ratio, 0.0)
                        box["width"] /= ratio
                        box["height"] /= ratio
                        if self._handler is not None:
                            self._handler(FireSmokeEvent("fire_smoke_detection", request.stream_id, request.frame_id, status, confidence, box))
                    self._metrics.processed += 1
                    self._metrics.record_task_latency(
                        (time.monotonic() - request.submitted_at_monotonic) * 1000.0
                    )
                except Exception as exc:
                    self._error = str(exc)
                    self._metrics.errors += 1
                    logging.exception("fire/smoke task failed")
