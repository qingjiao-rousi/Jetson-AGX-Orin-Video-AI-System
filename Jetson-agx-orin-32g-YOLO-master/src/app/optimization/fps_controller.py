from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class FpsController:
    """自适应帧率控制 — 根据 GPU 负载 + 队列深度做丢帧决策。

    策略:
      - GPU > 85% 或 queue_depth > 80%  → drop_rate 逐步增加
      - GPU < 60% 且 queue_depth < 40%  → drop_rate 逐步恢复
      - 每次 observe() 更新滑动窗口统计，不阻塞热路径。
    """

    _settings: object
    _window: deque = field(default_factory=lambda: deque(maxlen=100), init=False, repr=False)
    _drop_counter: int = field(default=0, init=False, repr=False)
    _total_frames: int = field(default=0, init=False, repr=False)
    _current_drop_rate: float = field(default=0.0, init=False, repr=False)
    _last_drop_decision: bool = field(default=False, init=False, repr=False)

    # 阈值（可从 settings 覆盖）
    _gpu_high: float = 85.0
    _gpu_low: float = 60.0
    _queue_high: float = 0.80
    _queue_low: float = 0.40
    _drop_step_up: float = 0.15
    _drop_step_down: float = 0.05
    _min_drop_rate: float = 0.0
    _max_drop_rate: float = 0.90

    # ──── 热路径：每帧调用 — 必须轻量 ────
    def observe(self, result: object) -> bool:
        """返回 True 表示建议丢弃当前帧。"""
        self._total_frames += 1

        gpu_pct = self._read_gpu()
        queue_pct = self._read_queue_depth()

        self._window.append((gpu_pct, queue_pct))

        prev_rate = self._current_drop_rate

        if gpu_pct > self._gpu_high or queue_pct > self._queue_high:
            self._current_drop_rate = min(
                self._max_drop_rate,
                self._current_drop_rate + self._drop_step_up,
            )
        elif gpu_pct < self._gpu_low and queue_pct < self._queue_low:
            self._current_drop_rate = max(
                self._min_drop_rate,
                self._current_drop_rate - self._drop_step_down,
            )
        # 否则保持当前 drop_rate

        # 概率丢帧（避免固定模式导致抖动）
        import random

        drop = random.random() < self._current_drop_rate
        if drop:
            self._drop_counter += 1

        self._last_drop_decision = drop
        return drop

    # ──── GPU 读取 ────
    def _read_gpu(self) -> float:
        """从 GPU 监控器读取当前利用率。未接入真机时返回安全默认值。"""
        monitor = getattr(self, "_gpu_monitor", None)
        if monitor is not None and hasattr(monitor, "gpu_util"):
            val = monitor.gpu_util()
            if isinstance(val, (int, float)):
                return float(val)
        # 离线模式：返回中等负载，不触发丢帧
        return 50.0

    # ──── 队列深度 ────
    def _read_queue_depth(self) -> float:
        """估算 pipeline 内部队列深度 (0.0 ~ 1.0)。"""
        bp = getattr(self, "_backpressure", None)
        if bp is not None and hasattr(bp, "queue_depth_ratio"):
            return float(bp.queue_depth_ratio())
        return 0.3  # 默认 30%

    # ──── 绑定外部监控 ────
    def bind_gpu_monitor(self, monitor: object) -> None:
        self._gpu_monitor = monitor

    def bind_backpressure(self, controller: object) -> None:
        self._backpressure = controller

    # ──── 统计 ────
    def stats(self) -> dict[str, object]:
        recent = list(self._window)
        avg_gpu = sum(g for g, _ in recent) / max(len(recent), 1)
        avg_queue = sum(q for _, q in recent) / max(len(recent), 1)
        return {
            "enabled": getattr(self._settings, "enable_fps_control", True),
            "total_frames": self._total_frames,
            "dropped_frames": self._drop_counter,
            "drop_ratio": self._drop_counter / max(self._total_frames, 1),
            "current_drop_rate": round(self._current_drop_rate, 3),
            "last_drop_decision": self._last_drop_decision,
            "avg_gpu_util": round(avg_gpu, 1),
            "avg_queue_depth": round(avg_queue, 3),
            "window_size": len(recent),
        }