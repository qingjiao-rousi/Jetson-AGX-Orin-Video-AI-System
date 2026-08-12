from __future__ import annotations

"""主 nvinfer 前的自适应 FPS gate，目标是在过载时优先保持结果新鲜度。"""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class FpsController:
    """自适应帧率控制器，根据 GPU 负载和主结果写出背压做丢帧决策。

    策略:
      - GPU > 85% 或 queue_depth > 80%  → drop_rate 逐步增加
      - GPU < 60% 且 queue_depth < 40%  → drop_rate 逐步恢复
      - 每次 ``should_drop_frame()`` 位于 probe 热路径，保持为常数级读数和随机决策。

    该控制器的队列信号来自主结果 writer 背压，不覆盖专用 worker/FrameStore 积压；
    专用任务 freshness 由各自任务队列与 stale deadline 管理。
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
        """兼容旧调用点；当前实际 gate 在 nvinfer sink probe 调用 ``should_drop_frame``。"""
        return self.should_drop_frame()

    def should_drop_frame(self) -> bool:
        """在进入主 nvinfer 前决定是否丢弃当前 buffer；True 会由 pad probe 返回 DROP。"""
        if not getattr(self._settings, "enable_fps_control", True):
            self._last_drop_decision = False
            return False
        # 计数发生在 gate 处，包含最终保留和被 DROP 的 buffer，用于解释控制器自身丢帧率。
        self._total_frames += 1

        gpu_pct = self._read_gpu()
        queue_pct = self._read_queue_depth()

        self._window.append((gpu_pct, queue_pct))

        prev_rate = self._current_drop_rate

        # 高水位快速加大丢帧，低水位缓慢恢复，形成滞后以避免在阈值附近来回抖动。
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

        # 概率丢帧避免固定间隔造成周期性抖动；因此单次运行的丢帧序列不可逐帧复现。
        import random

        drop = random.random() < self._current_drop_rate
        if drop:
            self._drop_counter += 1

        self._last_drop_decision = drop
        return drop

    # ──── GPU 读取 ────
    def _read_gpu(self) -> float:
        """读取最近 GR3D 利用率；离线默认中负载只为避免单测/无硬件环境触发丢帧。"""
        monitor = getattr(self, "_gpu_monitor", None)
        if monitor is not None and hasattr(monitor, "gpu_util"):
            val = monitor.gpu_util()
            if isinstance(val, (int, float)):
                return float(val)
        # 离线模式：返回中等负载，不触发丢帧
        return 50.0

    # ──── 队列深度 ────
    def _read_queue_depth(self) -> float:
        """读取主结果背压比例 (0~1)，不是 GStreamer 内部全部 queue 的真实深度。"""
        bp = getattr(self, "_backpressure", None)
        if bp is not None and hasattr(bp, "queue_depth_ratio"):
            return float(bp.queue_depth_ratio())
        return 0.3  # 默认 30%

    # ──── 绑定外部监控 ────
    def bind_gpu_monitor(self, monitor: object) -> None:
        """注入可选硬件监控器，保持控制器可在单测中独立创建。"""
        self._gpu_monitor = monitor

    def bind_backpressure(self, controller: object) -> None:
        self._backpressure = controller

    # ──── 统计 ────
    def stats(self) -> dict[str, object]:
        """返回最近窗口的决策与输入信号，供 benchmark 解释丢帧。"""
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
