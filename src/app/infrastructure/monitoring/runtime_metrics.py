from __future__ import annotations

"""运行时指标聚合器：逐帧状态、队列/控制快照、资源快照和有限时延样本。"""

import json
import os
import resource
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

# 使用 FrameResult 的 stream/frame 作为时延匹配键，避免把多路 batch 结果混为一个样本。
from app.domain.entities import FrameResult


class RuntimeMetricsRecorder:
    """采集可复现实验和 dashboard 共用的运行时快照。

    时延定义严格为 ``primary-infer:sink`` 到对应主结果 JSONL 成功写入，使用单调时钟。
    它覆盖主 pipeline、Python 编排和 writer 排队，但不包含相机采集、解码前等待、
    显示、编码或网络传输，不能表述为 camera-to-display 延迟。
    """
    def __init__(
        self,
        path: Path | None,
        *,
        interval_seconds: float = 1.0,
        stale_after_seconds: float = 5.0,
        enable_last_frame_keepalive: bool = True,
        keepalive_timeout_ms: int = 1000,
    ) -> None:
        self._path = path
        self._interval_seconds = max(interval_seconds, 0.1)
        self._stale_after_seconds = max(stale_after_seconds, 0.1)
        self._enable_last_frame_keepalive = enable_last_frame_keepalive
        self._keepalive_timeout_seconds = max(keepalive_timeout_ms / 1000.0, 0.1)
        self._lock = Lock()
        self._file = None
        self._started_at = 0.0
        self._last_emit_at = 0.0
        self._last_gpu_snapshot: dict[str, object] = {}
        self._total_frames = 0
        self._streams: dict[str, dict[str, Any]] = {}
        self._probe_metrics_provider = None
        self._control_metrics_provider = None
        self._queue_metrics_provider = None
        # 两张待匹配表连接 probe 起点、编排器结果与异步 JSONL 写入三个时间点。
        self._pending_pipeline: dict[tuple[str, int], float] = {}
        self._pending_result_write: dict[tuple[str, int], float] = {}
        self._pipeline_latency_ms: deque[float] = deque(maxlen=20_000)
        self._writer_latency_ms: deque[float] = deque(maxlen=20_000)
        self._end_to_end_latency_ms: deque[float] = deque(maxlen=20_000)
        self._unmatched_results = 0
        self._unmatched_writes = 0
        self._evicted_pending_starts = 0
        self._evicted_pending_results = 0
        self._pending_limit = 20_000

    def set_probe_metrics_provider(self, provider) -> None:
        """注册 C++/Python metadata probe 指标提供者，采样时按需读取。"""
        self._probe_metrics_provider = provider

    def set_control_metrics_provider(self, provider) -> None:
        """注册 FPS 与背压控制器指标提供者，避免控制器反向依赖 recorder。"""
        self._control_metrics_provider = provider

    def set_queue_metrics_provider(self, provider) -> None:
        """注册 writer/task-buffer/FrameStore/worker 队列快照，不耦合具体组件类型。"""
        self._queue_metrics_provider = provider

    def start(self) -> None:
        """开始新的进程内指标窗口；配置路径存在时以 JSONL 追加周期快照。"""
        self._started_at = time.monotonic()
        self._last_emit_at = 0.0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")

    def observe(self, result: FrameResult, *, gpu_snapshot: dict[str, object] | None = None) -> None:
        """记录一帧主结果、按流状态及 pipeline 到编排器的时延，并按周期落盘。"""
        now = time.monotonic()
        with self._lock:
            if gpu_snapshot:
                self._last_gpu_snapshot = dict(gpu_snapshot)
            # 主结果抵达此处时 pipeline 段结束，但 writer 段还未完成。
            self._record_result_latency_locked(result, now)
            self._total_frames += 1
            stream = self._streams.setdefault(
                result.stream_id,
                {
                    "stream_id": result.stream_id,
                    "frame_count": 0,
                    "first_seen_monotonic": now,
                    "last_seen_monotonic": now,
                    "last_frame_id": None,
                    "stale_count": 0,
                    "recovered_count": 0,
                    "status": "online",
                },
            )
            previous_status = stream.get("status")
            stream["frame_count"] += 1
            stream["last_seen_monotonic"] = now
            stream["last_frame_id"] = result.frame_id
            stream["last_seen_at"] = _utc_now()
            stream["last_keepalive_at"] = None
            stream["keepalive_active"] = False
            stream["detections"] = len(result.detections)
            stream["tracks"] = len(result.tracks)
            stream["status"] = "online"
            if previous_status == "stale":
                stream["recovered_count"] += 1

            # 指标按固定周期写，而非每帧写，避免 metrics I/O 自己成为性能变量。
            if now - self._last_emit_at >= self._interval_seconds:
                self._mark_stale_streams(now)
                self._emit_locked(now, gpu_snapshot=gpu_snapshot)

    def snapshot(self) -> dict[str, Any]:
        """返回即时快照并更新 stale 流状态，不强制写文件。"""
        now = time.monotonic()
        with self._lock:
            self._mark_stale_streams(now)
            return self._payload(now, gpu_snapshot=None)

    def close(self) -> None:
        """关闭前写出最后一条聚合快照，确保短时 benchmark 也有尾部指标。"""
        with self._lock:
            if self._started_at > 0:
                self._emit_locked(time.monotonic(), gpu_snapshot=self._last_gpu_snapshot)
            if self._file is not None and not self._file.closed:
                self._file.close()

    def mark_pipeline_start(self, stream_id: str, frame_id: int) -> None:
        """在主 nvinfer 之前标记时延起点。

        使用 monotonic 时钟衡量应用处理时间，不是摄像机采集到显示端的物理延迟。
        """
        key = (str(stream_id), int(frame_id))
        with self._lock:
            self._bounded_store_locked(self._pending_pipeline, key, time.monotonic(), "start")

    def mark_result_written(self, result: FrameResult) -> None:
        """在异步 writer 成功落盘后匹配同一帧，完成 writer 与端到端样本。"""
        key = (result.stream_id, int(result.frame_id))
        now = time.monotonic()
        with self._lock:
            result_started = self._pending_result_write.pop(key, None)
            if result_started is None:
                self._unmatched_writes += 1
                return
            self._writer_latency_ms.append((now - result_started) * 1000.0)
            # 仅在同一 stream/frame 的起点存在时才形成 end_to_end 样本。
            pipeline_started = self._pending_pipeline.pop(key, None)
            if pipeline_started is not None:
                self._end_to_end_latency_ms.append((now - pipeline_started) * 1000.0)

    def _record_result_latency_locked(self, result: FrameResult, now: float) -> None:
        """主结果抵达编排器时记录 pipeline 段，并保存等待 JSONL 落盘的中间标记。"""
        key = (result.stream_id, int(result.frame_id))
        pipeline_started = self._pending_pipeline.get(key)
        if pipeline_started is None:
            self._unmatched_results += 1
        else:
            self._pipeline_latency_ms.append((now - pipeline_started) * 1000.0)
        self._bounded_store_locked(self._pending_result_write, key, now, "result")

    def _bounded_store_locked(
        self,
        target: dict[tuple[str, int], float],
        key: tuple[str, int],
        value: float,
        kind: str,
    ) -> None:
        """限制未匹配帧的状态表大小；极端丢帧/写入失败不能导致 metrics 内存无界增长。"""
        if key not in target and len(target) >= self._pending_limit:
            oldest_key = next(iter(target))
            target.pop(oldest_key, None)
            if kind == "start":
                self._evicted_pending_starts += 1
            else:
                self._evicted_pending_results += 1
        target[key] = value

    def _mark_stale_streams(self, now: float) -> None:
        """按最后结果到达时间标记 stale；keepalive 是状态标志，不会生成新推理帧。"""
        for stream in self._streams.values():
            age = now - float(stream.get("last_seen_monotonic", now))
            if age <= self._stale_after_seconds:
                continue
            if stream.get("status") != "stale":
                stream["status"] = "stale"
                stream["stale_count"] = int(stream.get("stale_count", 0)) + 1
            stream["stale_seconds"] = round(age, 3)
            if self._enable_last_frame_keepalive and age >= self._keepalive_timeout_seconds:
                stream["keepalive_active"] = True
                stream["last_keepalive_at"] = _utc_now()

    def _emit_locked(self, now: float, *, gpu_snapshot: dict[str, object] | None) -> None:
        self._last_emit_at = now
        if self._file is None or self._file.closed:
            return
        self._file.write(json.dumps(self._payload(now, gpu_snapshot=gpu_snapshot), ensure_ascii=False) + "\n")
        self._file.flush()

    def _payload(self, now: float, *, gpu_snapshot: dict[str, object] | None) -> dict[str, Any]:
        """生成单条 JSONL 指标快照，所有 provider 在此刻读取最新有界状态。"""
        elapsed = max(now - self._started_at, 0.001)
        return {
            "timestamp": _utc_now(),
            "pid": os.getpid(),
            "elapsed_seconds": round(elapsed, 3),
            "total_frames": self._total_frames,
            "processing_fps": round(self._total_frames / elapsed, 3),
            "process": _process_snapshot(),
            "gpu": gpu_snapshot or self._last_gpu_snapshot,
            "probe": self._probe_metrics_provider() if self._probe_metrics_provider else {},
            "controls": self._control_metrics_provider() if self._control_metrics_provider else {},
            "queues": self._queue_metrics_provider() if self._queue_metrics_provider else {},
            "latency": {
                "definition": "primary_infer_sink_to_json_write_ms",
                "pipeline": _latency_summary(self._pipeline_latency_ms),
                "json_writer": _latency_summary(self._writer_latency_ms),
                "end_to_end": _latency_summary(self._end_to_end_latency_ms),
                "unmatched_results": self._unmatched_results,
                "unmatched_writes": self._unmatched_writes,
                "pending_pipeline_frames": len(self._pending_pipeline),
                "pending_writer_frames": len(self._pending_result_write),
                "evicted_pending_starts": self._evicted_pending_starts,
                "evicted_pending_results": self._evicted_pending_results,
            },
            "streams": {
                stream_id: _stream_payload(stream, now)
                for stream_id, stream in sorted(self._streams.items())
            },
        }


def _stream_payload(stream: dict[str, Any], now: float) -> dict[str, Any]:
    """生成分路吞吐/状态估算；dropped_frame_rate 由 frame_id 间隙推算，并非解码器硬计数。"""
    first_seen = float(stream.get("first_seen_monotonic", now))
    last_seen = float(stream.get("last_seen_monotonic", now))
    elapsed = max(last_seen - first_seen, 0.001)
    age = now - last_seen
    frame_count = int(stream.get("frame_count", 0))
    last_frame_id = stream.get("last_frame_id")
    frame_span = max(int(last_frame_id) + 1, frame_count) if last_frame_id is not None else frame_count
    dropped_frames = max(frame_span - frame_count, 0)
    return {
        "stream_id": stream.get("stream_id"),
        "status": stream.get("status", "unknown"),
        "frame_count": frame_count,
        "last_frame_id": last_frame_id,
        "last_seen_at": stream.get("last_seen_at"),
        "last_keepalive_at": stream.get("last_keepalive_at"),
        "keepalive_active": bool(stream.get("keepalive_active", False)),
        "last_seen_age_seconds": round(age, 3),
        "frame_age_ms": round(max(age, 0.0) * 1000, 1),
        "estimated_processing_fps": round(int(stream.get("frame_count", 0)) / elapsed, 3),
        "detections": int(stream.get("detections", 0)),
        "detection_count": int(stream.get("detections", 0)),
        "tracks": int(stream.get("tracks", 0)),
        "track_count": int(stream.get("tracks", 0)),
        "dropped_frames": dropped_frames,
        "dropped_frame_rate": round(dropped_frames / max(frame_span, 1), 4),
        "stale_count": int(stream.get("stale_count", 0)),
        "recovered_count": int(stream.get("recovered_count", 0)),
    }


def _process_snapshot() -> dict[str, Any]:
    """采集当前进程 CPU 时间和峰值 RSS；不等同全系统 CPU/RAM 使用量。"""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_seconds": round(float(usage.ru_utime), 3),
        "system_cpu_seconds": round(float(usage.ru_stime), 3),
        "max_rss_kb": int(usage.ru_maxrss),
    }


def _latency_summary(samples: deque[float]) -> dict[str, float | int | None]:
    """基于有限历史窗口汇总时延，长时间运行时早期样本会被淘汰。"""
    values = sorted(samples)
    if not values:
        return {"samples": 0, "average_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "samples": len(values),
        "average_ms": round(sum(values) / len(values), 3),
        "p50_ms": round(_percentile(values, 50), 3),
        "p95_ms": round(_percentile(values, 95), 3),
        "max_ms": round(values[-1], 3),
    }


def _percentile(values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile for a finite in-memory sample set."""
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
