from __future__ import annotations

"""姿态专用模型的 batch=1 异步 worker 与关键点后处理。"""

from dataclasses import dataclass
import logging
from threading import Event, Thread, current_thread
import time
from typing import Any

import numpy as np

# 复用通用 TensorRT backend 与 ROI 工具；类名保留历史名称，不代表姿态使用 PPE 模型。
from app.application.helmet_service import TensorRTHelmetBackend, crop_person_roi, letterbox
from app.application.routing_policy import TaskRequest
from app.application.task_metrics import TaskExecutionMetrics


@dataclass(frozen=True)
class PoseEvent:
    """一条 person ROI 的 17 个关键点观测，不在此层推导动作或跌倒结论。"""
    event_type: str
    stream_id: str
    track_id: int
    frame_id: int
    keypoints: tuple[tuple[float, float, float], ...]
    confidence: float


def decode_pose_output(output: np.ndarray, roi_shape: tuple[int, int], input_size: int = 640) -> tuple[tuple[float, float, float], ...] | None:
    """选择最高置信人体候选，并把 COCO-17 keypoints 从 letterbox 坐标映射回人员 ROI。"""
    raw = np.asarray(output)
    if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
        raw = raw.transpose(0, 2, 1)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] < 56:
        return None
    # 当前仅保留最高 objectness 的一个 person；路由已按主 tracker 的 person ROI 调度。
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
    """独立消费 pose 队列的 batch=1 worker。

    当前 pose engine/后处理按单 ROI 调用，未参与 PPE 微批实验；worker 的独立线程
    使其积压和错误不会阻塞 DeepStream probe 或其它专用任务。
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
        self._processed = 0
        self._emitted = 0
        self._metrics = TaskExecutionMetrics()
        self._logged_output = False

    def set_event_handler(self, handler) -> None:
        self._handler = handler

    def start(self) -> None:
        """启动姿态线程；首次有任务时才反序列化 TensorRT engine。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="pose-task-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """有界停止 worker，避免阻塞应用总停机流程。"""
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
        """按请求执行：取帧 -> 人员 ROI -> 预处理 -> 推理 -> 关键点事件。"""
        while not self._stop_event.wait(0.02):
            # 显式 batch=1：当前配置/engine 不应被 PPE 的微批策略强行复用。
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
                # frame_id 精确匹配失败意味着缓存窗口已淘汰，直接计数而不使用“最新帧”冒充原帧。
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
                    # 仅首次记录实际 engine 输出布局，便于部署时诊断导出模型不匹配。
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
