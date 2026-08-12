from __future__ import annotations

"""以主结果 writer 的生产/消费差估算背压，为 FPS gate 提供轻量反馈。"""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class BackpressureController:
    """主结果写出背压控制器，监控下游 JSONL 消费速率。

    策略:
      - 记录每帧的 stream_id/frame_id，跟踪 timestamp 增量
      - 如果 produce 速率远超 consume（窗口内积压超过阈值），触发背压信号
      - 背压信号供 FpsController 参考，本类本身不直接丢帧。

    ``pending_count`` 是最近有界生产/消费窗口的差值，近似 writer 队列压力而不是
    GStreamer、FrameStore 或全部 worker 队列的总积压。
    """

    _settings: object
    _produce_timestamps: deque = field(default_factory=lambda: deque(maxlen=500), init=False, repr=False)
    _consume_timestamps: deque = field(default_factory=lambda: deque(maxlen=500), init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)
    _max_pending: int = field(default=0, init=False, repr=False)
    _backpressure_active: bool = field(default=False, init=False, repr=False)
    _observations: int = field(default=0, init=False, repr=False)
    _last_result: object = field(default=None, init=False, repr=False)

    # 阈值
    _queue_limit_ratio: float = 0.75   # pending / max_queue_size ≥ 75% → backpressure
    _release_ratio: float = 0.30       # 降到 30% 以下 → 解除

    # ──── 热路径：每帧调用 ────
    def observe(self, result: object) -> None:
        """在编排器收到一个主结果时记录生产；对应消费在 JSONL 成功写入后标记。"""
        if not getattr(self._settings, "enable_backpressure", True):
            self._backpressure_active = False
            return
        self._observations += 1
        self._last_result = result

        now = time.monotonic()
        # 生产端是 Orchestrator 收到 FrameResult；消费端仅在 JsonWriter flush 成功后推进。
        self._produce_timestamps.append(now)
        self._pending_count = max(0, len(self._produce_timestamps) - len(self._consume_timestamps))

        max_q = getattr(self._settings, "max_queue_size", 32)
        self._max_pending = max(self._max_pending, self._pending_count)

        # 背压判定
        if self._pending_count >= int(max_q * self._queue_limit_ratio):
            self._backpressure_active = True
        elif self._pending_count <= int(max_q * self._release_ratio):
            self._backpressure_active = False

    def mark_consumed(self) -> None:
        """由 JsonWriter 成功落盘回调调用，减少 writer 侧的估算积压。"""
        if not getattr(self._settings, "enable_backpressure", True):
            return
        # writer 回调可能滞后于 producer；窗口差仅作为实时控制信号，不是持久队列审计。
        self._consume_timestamps.append(time.monotonic())
        self._pending_count = max(0, len(self._produce_timestamps) - len(self._consume_timestamps))

        if self._pending_count <= int(getattr(self._settings, "max_queue_size", 32) * self._release_ratio):
            self._backpressure_active = False

    # ──── 查询 ────
    @property
    def is_active(self) -> bool:
        return self._backpressure_active

    def queue_depth_ratio(self) -> float:
        """返回估算 writer 压力比例 (0~1)，供 FPS gate 参考。"""
        if not getattr(self._settings, "enable_backpressure", True):
            return 0.0
        max_q = max(getattr(self._settings, "max_queue_size", 32), 1)
        return min(1.0, self._pending_count / max_q)

    # ──── 统计 ────
    def stats(self) -> dict[str, object]:
        """返回背压状态和最近结果标识，供运行报告定位写出瓶颈。"""
        return {
            "enabled": getattr(self._settings, "enable_backpressure", True),
            "observations": self._observations,
            "pending_count": self._pending_count,
            "max_pending_ever": self._max_pending,
            "backpressure_active": self._backpressure_active,
            "queue_depth_ratio": round(self.queue_depth_ratio(), 3),
            "queue_limit": getattr(self._settings, "max_queue_size", 32),
            "last_stream_id": getattr(self._last_result, "stream_id", None),
            "last_frame_id": getattr(self._last_result, "frame_id", None),
        }
