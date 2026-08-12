from __future__ import annotations

"""专用模型 worker 的有界时延与异常计数，供运行时 metrics 汇总。"""

from collections import deque
from threading import Lock
from typing import Iterable


def sample_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    """汇总有限样本的均值/P50/P95，避免长运行服务无界保留每次任务数据。"""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"samples": 0, "average": None, "p50": None, "p95": None, "max": None}
    return {
        "samples": len(ordered),
        "average": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "max": round(ordered[-1], 3),
    }


class TaskExecutionMetrics:
    """所有非主模型 worker 共享的执行指标容器。

    ``queue_wait_ms`` 是请求提交到 worker drain 的等待；``inference_ms`` 是模型调用
    区间；``task_latency_ms`` 是 worker 侧完整任务耗时。三者均不包含主 pipeline
    解码/推理时延，不能直接替代端到端时延。
    """

    def __init__(self, sample_limit: int = 2048) -> None:
        self.processed = 0
        self.missing_frames = 0
        self.errors = 0
        self._lock = Lock()
        self._queue_wait_ms: deque[float] = deque(maxlen=sample_limit)
        self._inference_ms: deque[float] = deque(maxlen=sample_limit)
        self._task_latency_ms: deque[float] = deque(maxlen=sample_limit)

    def record_queue_wait(self, value_ms: float) -> None:
        """记录任务在 TaskRequestBuffer 中等待的时间。"""
        with self._lock:
            self._queue_wait_ms.append(max(float(value_ms), 0.0))

    def record_inference(self, value_ms: float) -> None:
        """记录 TensorRT 调用边界内的时间，不含 ROI 读取与事件写出。"""
        with self._lock:
            self._inference_ms.append(max(float(value_ms), 0.0))

    def record_task_latency(self, value_ms: float) -> None:
        """记录 worker 从取得请求到产生结果的完整任务时延。"""
        with self._lock:
            self._task_latency_ms.append(max(float(value_ms), 0.0))

    def stats(self) -> dict[str, object]:
        """返回线程安全快照；deque 最大长度限制了分位数统计的历史窗口。"""
        with self._lock:
            return {
                "processed": self.processed,
                "missing_frames": self.missing_frames,
                "errors": self.errors,
                "queue_wait_ms": sample_summary(tuple(self._queue_wait_ms)),
                "inference_ms": sample_summary(tuple(self._inference_ms)),
                "task_latency_ms": sample_summary(tuple(self._task_latency_ms)),
            }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
