from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class BackpressureController:
    """背压保护 — 监控下游消费速率，防止 pipeline 内部堆积。

    策略:
      - 记录每帧的 stream_id/frame_id，跟踪 timestamp 增量
      - 如果 produce 速率远超 consume（窗口内积压超过阈值），触发背压信号
      - 背压信号供 FpsController 参考，不直接丢帧
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
        """生产者端：每产生一个 FrameResult 调用一次。"""
        self._observations += 1
        self._last_result = result

        now = time.monotonic()
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
        """消费者端：每次写出/处理完一条结果调用一次。"""
        self._consume_timestamps.append(time.monotonic())
        self._pending_count = max(0, len(self._produce_timestamps) - len(self._consume_timestamps))

        if self._pending_count <= int(getattr(self._settings, "max_queue_size", 32) * self._release_ratio):
            self._backpressure_active = False

    # ──── 查询 ────
    @property
    def is_active(self) -> bool:
        return self._backpressure_active

    def queue_depth_ratio(self) -> float:
        """返回当前队列深度比例 (0.0 ~ 1.0)，供 FpsController 参考。"""
        max_q = max(getattr(self._settings, "max_queue_size", 32), 1)
        return min(1.0, self._pending_count / max_q)

    # ──── 统计 ────
    def stats(self) -> dict[str, object]:
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