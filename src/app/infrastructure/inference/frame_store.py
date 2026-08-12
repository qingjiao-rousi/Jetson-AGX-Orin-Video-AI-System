from __future__ import annotations

"""在 DeepStream probe 与异步专用 worker 之间传递有限数量的源帧副本。"""

from collections import OrderedDict, deque
from threading import Lock
import time
from typing import Iterable

# cv2 只在 worker 读取时完成 RGBA->BGR；probe 写入保持 surface 拷贝的原始通道。
import cv2
import numpy as np

from app.domain.entities import canonical_stream_id


class FrameStore:
    """面向 ROI 专用任务的有界 CPU 帧缓存。

    probe 线程从 NVMM surface 复制 numpy 帧后立即返回；worker 不能持有原生 surface，
    只能用主检测的 ``stream_id/frame_id`` 查询这里的副本。全局 LRU 上限保护总内存，
    可选 per-stream 上限防止高帧率分路独占缓存。
    """

    def __init__(
        self,
        capture_stream_ids: Iterable[str] = (),
        max_size: int = 128,
        max_per_stream: int | None = None,
    ) -> None:
        # capture 集合由 bootstrap 根据 source.capabilities 推导，而非由 worker 自行决定。
        self._capture_stream_ids = {canonical_stream_id(value) for value in capture_stream_ids}
        self._max_size = max(int(max_size), 1)
        self._max_per_stream = (
            max(int(max_per_stream), 1) if max_per_stream is not None else None
        )
        self._lock = Lock()
        self._order: OrderedDict[tuple[str, int], None] = OrderedDict()
        self._order_by_stream: dict[str, OrderedDict[tuple[str, int], None]] = {}
        self._frames: dict[tuple[str, int], tuple[np.ndarray, float, int]] = {}
        self._evicted = 0
        self._puts = 0
        self._pending_bytes = 0
        self._evicted_by_stream: dict[str, int] = {}
        self._evicted_global = 0
        self._evicted_per_stream = 0
        self._pending_by_stream: dict[str, int] = {}
        self._consumer_hits: dict[str, int] = {}
        self._consumer_misses: dict[str, int] = {}
        self._consumer_frame_age_ms: dict[str, deque[float]] = {}

    def should_capture(self, stream_id: str) -> bool:
        """仅为配置了专用能力的流启用 CPU 拷贝，基础检测流无需进入缓存。"""
        return canonical_stream_id(stream_id) in self._capture_stream_ids

    def put(self, stream_id: str, frame_id: int, frame: np.ndarray) -> None:
        """写入一个三通道或四通道帧副本，并在容量满时先淘汰最旧帧。

        这里必须复制数组：pyds surface 与 GStreamer buffer 的所有权仍在主 pipeline，
        离开 probe 回调后不能把 numpy view 交给 worker。
        """
        key = (canonical_stream_id(stream_id), int(frame_id))
        image = np.asarray(frame)
        if image.ndim != 3:
            return
        with self._lock:
            self._puts += 1
            # 同 frame_id 覆盖时保持顺序位置，仅更新副本/内存计数。
            if key in self._frames:
                previous = self._frames[key]
                copied = image.copy()
                self._pending_bytes += int(copied.nbytes) - previous[2]
                self._frames[key] = (copied, time.monotonic(), int(copied.nbytes))
                return
            stream_order = self._order_by_stream.setdefault(key[0], OrderedDict())
            # 可选的分路上限先执行，保证某一路爆发不会耗尽整个共享窗口。
            while (
                self._max_per_stream is not None
                and self._pending_by_stream.get(key[0], 0) >= self._max_per_stream
            ):
                old_key = next(iter(stream_order))
                self._evict_locked(old_key, reason="per_stream")
            # 全局 OrderedDict 按插入顺序淘汰，缓存语义是“最近窗口”而非持久帧仓库。
            while len(self._frames) >= self._max_size:
                old_key = next(iter(self._order))
                self._evict_locked(old_key, reason="global")
            copied = image.copy()
            self._frames[key] = (copied, time.monotonic(), int(copied.nbytes))
            self._pending_bytes += int(copied.nbytes)
            self._pending_by_stream[key[0]] = self._pending_by_stream.get(key[0], 0) + 1
            self._order[key] = None
            stream_order[key] = None

    def get(self, stream_id: str, frame_id: int, *, consumer: str | None = None) -> np.ndarray | None:
        """按主检测帧号读取副本，并再复制一次交给消费者避免其原地修改缓存。"""
        key = (canonical_stream_id(stream_id), int(frame_id))
        with self._lock:
            record = self._frames.get(key)
            if record is None:
                self._record_consumer_miss(consumer)
                return None
            self._record_consumer_hit(consumer, (time.monotonic() - record[1]) * 1000.0)
            # 返回第二个副本；worker 预处理不可修改共享缓存，以免影响其它专用模型。
            return record[0].copy()

    def get_bgr(self, stream_id: str, frame_id: int, *, consumer: str | None = None) -> np.ndarray | None:
        """将 probe 处取得的 RGBA surface 转为专用模型常用的 BGR；非 RGBA 保持原样。"""
        frame = self.get(stream_id, frame_id, consumer=consumer)
        if frame is None:
            return None
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame

    def stats(self) -> dict[str, object]:
        """返回容量、内存、淘汰原因及各 worker 命中/缺帧/freshness 的指标快照。"""
        with self._lock:
            consumers = set(self._consumer_hits) | set(self._consumer_misses) | set(self._consumer_frame_age_ms)
            return {
                "puts": self._puts,
                "pending_frames": len(self._frames),
                "pending_bytes": self._pending_bytes,
                "max_size": self._max_size,
                "max_per_stream": self._max_per_stream,
                "pending_by_stream": dict(sorted(self._pending_by_stream.items())),
                "evicted": self._evicted,
                "dropped": self._evicted,
                "evicted_by_stream": dict(sorted(self._evicted_by_stream.items())),
                "evicted_global": self._evicted_global,
                "evicted_per_stream": self._evicted_per_stream,
                "by_consumer": {
                    name: {
                        "hits": self._consumer_hits.get(name, 0),
                        "misses": self._consumer_misses.get(name, 0),
                        "frame_age_ms": _sample_summary(self._consumer_frame_age_ms.get(name, ())),
                    }
                    for name in sorted(consumers)
                },
            }

    def _record_consumer_hit(self, consumer: str | None, age_ms: float) -> None:
        """frame_age 是 worker 取帧时的缓存年龄，不等同于完整端到端事件时延。"""
        if consumer is None:
            return
        self._consumer_hits[consumer] = self._consumer_hits.get(consumer, 0) + 1
        self._consumer_frame_age_ms.setdefault(consumer, deque(maxlen=2048)).append(max(age_ms, 0.0))

    def _record_consumer_miss(self, consumer: str | None) -> None:
        if consumer is None:
            return
        self._consumer_misses[consumer] = self._consumer_misses.get(consumer, 0) + 1

    def _evict_locked(self, key: tuple[str, int], *, reason: str) -> None:
        """同步删除全局和分路顺序索引，并分别累计全局/分路容量造成的淘汰。"""
        previous = self._frames.pop(key, None)
        self._order.pop(key, None)
        stream_order = self._order_by_stream.get(key[0])
        if stream_order is not None:
            stream_order.pop(key, None)
        if previous is None:
            return
        self._pending_bytes -= previous[2]
        self._pending_by_stream[key[0]] = self._pending_by_stream.get(key[0], 1) - 1
        if self._pending_by_stream[key[0]] <= 0:
            del self._pending_by_stream[key[0]]
        self._evicted += 1
        self._evicted_by_stream[key[0]] = self._evicted_by_stream.get(key[0], 0) + 1
        if reason == "per_stream":
            self._evicted_per_stream += 1
        else:
            self._evicted_global += 1


def _sample_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """对有界样本计算延迟分位数，避免为运行数小时的服务保留无界历史。"""
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


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
