from __future__ import annotations

"""把主检测的逐帧结果转换为按场景、能力和目标稳定性筛选的专用任务请求。"""

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
import time
from typing import Any

# 排队等待的 P50/P95 以有界样本汇总，避免任务运行越久内存越大。
from app.application.task_metrics import sample_summary

# Request 只复制主结果中 worker 必需的 ROI/轨迹信息，不携带完整 FrameResult 或图像。
from app.domain.entities import BoundingBox, FrameResult, Track, canonical_stream_id


@dataclass(frozen=True)
class TaskRequest:
    """一次专用推理的轻量请求，而非图像或模型执行对象。

    请求保存主检测时刻的 ROI、track 和调度上下文；worker 随后用
    ``(stream_id, frame_id)`` 从 FrameStore 取得实际帧。``submitted_at_monotonic``
    只用于进程内排队时延和陈旧判定，不能写成跨进程的事件时间。
    """

    task_name: str
    model_name: str
    stream_id: str
    source_name: str
    scene: str
    capability: str
    frame_id: int
    track_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    priority: str
    zones: tuple[str, ...] = ()
    submitted_at_monotonic: float = field(default_factory=time.monotonic, compare=False)


@dataclass
class _TrackTaskState:
    """同一 stream/task/track 的去重、稳定帧与间隔控制状态。"""
    last_seen_frame: int = -1
    stable_frames: int = 0
    last_submitted_frame: int = -1


class RoutingPolicy:
    """按 source 能力路由主检测结果到专用模型任务。

    该层只回答“某帧是否值得提交”。它不持有图像、不排队也不执行推理：队列容量、
    最新请求替换和 stale deadline 由 ``TaskRequestBuffer`` 管理。
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._lock = Lock()
        self._states: dict[tuple[str, str, int], _TrackTaskState] = {}
        self._submitted = 0
        self._filtered = 0
        self._unknown_streams = 0
        self._profiles = self._build_profiles(settings)
        self._capabilities = {
            item.name: item
            for item in getattr(settings, "capabilities", ())
            if item.enabled
        }
        self._tasks = {
            item.name: item
            for item in getattr(settings, "model_tasks", ())
            if item.enabled
        }
        self._models = {
            item.name: item
            for item in getattr(settings, "models", ())
            if item.enabled
        }

    def route(self, result: FrameResult) -> tuple[TaskRequest, ...]:
        """为一帧结果生成去重后的任务请求。

        一路可配置多个 capability，但相同 task 在同一帧最多提交一次。目标触发任务
        遍历 tracker 轨迹；帧触发任务使用整帧虚拟 ROI，适合火焰/烟雾等非目标任务。
        """
        # 以 source profile 限制能力：同一 person 在生产区和车辆入口不会触发相同任务集合。
        profile = self._profile_for(result.stream_id)
        if profile is None:
            with self._lock:
                self._unknown_streams += 1
            return ()

        requests: list[TaskRequest] = []
        capability_names = tuple(getattr(profile, "capabilities", ()))
        # 能力可重叠，按 task 去重避免同一 PPE/姿态任务被重复投递。
        seen_tasks: set[str] = set()
        for capability_name in capability_names:
            capability = self._capabilities.get(capability_name)
            if capability is None:
                continue
            for task_name in capability.tasks:
                if task_name in seen_tasks:
                    continue
                seen_tasks.add(task_name)
                task = self._tasks.get(task_name)
                if task is None or task.model not in self._models:
                    continue
                # 烟火等整帧任务没有 person/car ROI，使用虚拟框但仍走同一队列契约。
                if task.frame_trigger:
                    if self._should_submit_frame(result, task_name, task):
                        requests.append(
                            TaskRequest(
                                task_name=task.name,
                                model_name=task.model,
                                stream_id=canonical_stream_id(result.stream_id),
                                source_name=profile.name,
                                scene=profile.scene,
                                capability=capability_name,
                                frame_id=result.frame_id,
                                track_id=1,
                                class_name="frame",
                                confidence=1.0,
                                bbox=BoundingBox(0.0, 0.0, 1280.0, 720.0),
                                priority=profile.priority,
                                zones=tuple(profile.zones),
                            )
                        )
                    continue
                for track in result.tracks:
                    if not self._matches_trigger(task, track):
                        continue
                    if self._should_submit(result, profile, capability_name, task_name, task, track):
                        requests.append(
                            TaskRequest(
                                task_name=task.name,
                                model_name=task.model,
                                stream_id=canonical_stream_id(result.stream_id),
                                source_name=profile.name,
                                scene=profile.scene,
                                capability=capability_name,
                                frame_id=result.frame_id,
                                track_id=track.track_id,
                                class_name=track.class_name,
                                confidence=track.confidence,
                                bbox=track.bbox,
                                priority=profile.priority,
                                zones=tuple(profile.zones),
                            )
                        )
        with self._lock:
            self._submitted += len(requests)
        return tuple(requests)

    def _should_submit_frame(self, result: FrameResult, task_name: str, task: Any) -> bool:
        """按帧任务仅受 interval 限流，不依赖 person/car 等 tracker 稳定性。"""
        state_key = (canonical_stream_id(result.stream_id), task_name, -1)
        with self._lock:
            # frame trigger 使用 track_id=-1 的独立状态，防止和真实目标轨迹状态混淆。
            state = self._states.setdefault(state_key, _TrackTaskState())
            interval = max(int(task.interval), 1)
            if state.last_submitted_frame >= 0 and result.frame_id - state.last_submitted_frame < interval:
                self._filtered += 1
                return False
            state.last_submitted_frame = result.frame_id
            self._submitted += 1
            return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "filtered": self._filtered,
                "unknown_streams": self._unknown_streams,
                "tracked_states": len(self._states),
            }

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._submitted = 0
            self._filtered = 0
            self._unknown_streams = 0

    def _build_profiles(self, settings: Any) -> dict[str, Any]:
        """建立 streammux 序号、配置名和 canonical stream ID 到 source profile 的多重索引。"""
        profiles: dict[str, Any] = {}
        for index, source in enumerate(getattr(settings, "sources", ())):
            if not source.enabled:
                continue
            profiles[canonical_stream_id(index)] = source
            profiles[canonical_stream_id(source.name, default_index=index)] = source
            profiles[source.name] = source
        return profiles

    def _profile_for(self, stream_id: str) -> Any | None:
        canonical = canonical_stream_id(stream_id)
        return self._profiles.get(canonical) or self._profiles.get(str(stream_id))

    def _matches_trigger(self, task: Any, track: Track) -> bool:
        """空 trigger_classes 表示任何已跟踪类别均可触发该任务。"""
        triggers = tuple(str(value).lower() for value in task.trigger_classes)
        if not triggers:
            return True
        class_name = str(track.class_name).lower()
        return class_name in triggers

    def _should_submit(
        self,
        result: FrameResult,
        profile: Any,
        capability_name: str,
        task_name: str,
        task: Any,
        track: Track,
    ) -> bool:
        """基于有效轨迹、连续出现帧数和提交间隔判断目标任务是否应投递。

        ``stable_frames`` 只在相邻 frame_id 连续时增长，视频跳帧或 tracker 中断都会
        重新计数，避免瞬时误检立即触发较重的专用模型。
        """
        if track.track_id < 0 or track.track_id == 0xFFFFFFFFFFFFFFFF:
            with self._lock:
                self._filtered += 1
            return False
        state_key = (canonical_stream_id(result.stream_id), task_name, track.track_id)
        with self._lock:
            state = self._states.setdefault(state_key, _TrackTaskState())
            # 同一 frame 的重复 metadata 不能重复提交专用任务。
            if state.last_seen_frame == result.frame_id:
                return False
            if state.last_seen_frame >= 0 and result.frame_id == state.last_seen_frame + 1:
                state.stable_frames += 1
            else:
                state.stable_frames = 1
            state.last_seen_frame = result.frame_id
            if state.stable_frames < task.min_track_frames:
                self._filtered += 1
                return False
            interval = max(int(task.interval), 0)
            if state.last_submitted_frame >= 0 and interval > 0:
                if result.frame_id - state.last_submitted_frame < interval:
                    self._filtered += 1
                    return False
            state.last_submitted_frame = result.frame_id
            return True


class TaskRequestBuffer:
    """供专用 worker 消费的按任务隔离、有限且偏向最新数据的请求队列。

    键为 ``(stream, task, track)``：同一目标的新请求替换旧请求，而不是堆积多个
    已过期 ROI。不同 task 有独立容量和 stale deadline，因此车牌积压不会挤掉 PPE。
    """

    def __init__(self, max_size: int = 32, *, task_settings: tuple[Any, ...] = ()) -> None:
        self._max_size = max(int(max_size), 1)
        self._lock = Lock()
        self._order_by_task: dict[str, deque[tuple[str, str, int]]] = {}
        self._items: dict[tuple[str, str, int], TaskRequest] = {}
        self._dropped = 0
        self._dropped_by_task: dict[str, int] = {}
        self._stale_dropped_by_task: dict[str, int] = {}
        self._replaced_by_task: dict[str, int] = {}
        self._submitted_by_task: dict[str, int] = {}
        self._queue_wait_ms_by_task: dict[str, deque[float]] = {}
        # 未配置 task 仍可使用全局上限，已配置任务则完全隔离自己的容量/陈旧截止时间。
        self._queue_size_by_task = {
            str(task.name): max(int(task.queue_size), 1)
            for task in task_settings
            if getattr(task, "queue_size", None) is not None
        }
        self._stale_after_ms_by_task = {
            str(task.name): max(int(task.stale_after_ms), 1)
            for task in task_settings
            if getattr(task, "stale_after_ms", None) is not None
        }
        self._configured_task_names = {str(task.name) for task in task_settings}

    def submit(self, requests: tuple[TaskRequest, ...] | list[TaskRequest]) -> None:
        """提交请求；同 key 替换为最新帧，容量满时淘汰该任务最早进入队列的请求。"""
        with self._lock:
            for request in requests:
                self._submitted_by_task[request.task_name] = (
                    self._submitted_by_task.get(request.task_name, 0) + 1
                )
                key = (request.stream_id, request.task_name, request.track_id)
                order = self._order_by_task.setdefault(request.task_name, deque())
                if key in self._items:
                    # 保留原队列位置但替换 payload，worker 取得的始终是最新 frame_id/ROI。
                    self._items[key] = request
                    self._replaced_by_task[request.task_name] = (
                        self._replaced_by_task.get(request.task_name, 0) + 1
                    )
                    continue
                queue_size = self._queue_size_by_task.get(request.task_name, self._max_size)
                # 满队列淘汰本任务最早请求，不跨任务删除，保证慢车牌不会挤占 PPE。
                while len(order) >= queue_size:
                    old_key = order.popleft()
                    if old_key in self._items:
                        del self._items[old_key]
                        self._dropped += 1
                        task_name = old_key[1]
                        self._dropped_by_task[task_name] = self._dropped_by_task.get(task_name, 0) + 1
                self._items[key] = request
                order.append(key)

    def drain(self, limit: int | None = None, *, task_name: str | None = None) -> tuple[TaskRequest, ...]:
        """取出可执行请求；指定 task 时只消费该任务，避免 worker 间争用同一队列。"""
        with self._lock:
            count = len(self._items) if limit is None else max(int(limit), 0)
            if task_name is not None:
                return tuple(self._drain_task_locked(task_name, count))
            requests: list[TaskRequest] = []
            # 未指定 worker 时按任务名确定顺序 drain；实际 worker 始终传 task_name。
            for name in sorted(self._order_by_task):
                requests.extend(self._drain_task_locked(name, count - len(requests)))
                if len(requests) >= count:
                    break
            return tuple(requests)

    def stats(self) -> dict[str, int]:
        """输出每任务提交、替换、容量丢弃、陈旧丢弃和排队等待的可观测性快照。"""
        with self._lock:
            pending_by_task: dict[str, int] = {}
            for request in self._items.values():
                pending_by_task[request.task_name] = pending_by_task.get(request.task_name, 0) + 1
            return {
                "mode": "per_task_latest",
                "default_queue_size": self._max_size,
                "pending": len(self._items),
                "dropped": self._dropped,
                "by_task": {
                    task_name: {
                        "submitted": self._submitted_by_task.get(task_name, 0),
                        "replaced": self._replaced_by_task.get(task_name, 0),
                        "dropped": self._dropped_by_task.get(task_name, 0),
                        "stale_dropped": self._stale_dropped_by_task.get(task_name, 0),
                        "pending": pending_by_task.get(task_name, 0),
                        "queue_size": self._queue_size_by_task.get(task_name, self._max_size),
                        "stale_after_ms": self._stale_after_ms_by_task.get(task_name),
                        "queue_wait_ms": sample_summary(
                            self._queue_wait_ms_by_task.get(task_name, ())
                        ),
                    }
                    for task_name in sorted(
                        set(self._submitted_by_task)
                        | set(self._replaced_by_task)
                        | set(self._dropped_by_task)
                        | set(self._stale_dropped_by_task)
                        | set(pending_by_task)
                        | set(self._queue_wait_ms_by_task)
                        | set(self._queue_size_by_task)
                        | set(self._stale_after_ms_by_task)
                        | self._configured_task_names
                    )
                },
            }

    def has_pending(self, task_name: str | None = None) -> bool:
        with self._lock:
            if task_name is None:
                return bool(self._items)
            return any(item.task_name == task_name for item in self._items.values())

    def _drain_task_locked(self, task_name: str, limit: int) -> list[TaskRequest]:
        """在 worker 消费边界执行 stale 判定，确保排队期间过期的任务不会进入 GPU。"""
        requests: list[TaskRequest] = []
        order = self._order_by_task.get(task_name)
        if order is None:
            return requests
        now = time.monotonic()
        deadline_ms = self._stale_after_ms_by_task.get(task_name)
        while order and len(requests) < limit:
            key = order.popleft()
            request = self._items.pop(key, None)
            if request is None:
                continue
            # monotonic 时钟不受系统校时影响，适合衡量本进程中的真实排队等待。
            wait_ms = max((now - request.submitted_at_monotonic) * 1000.0, 0.0)
            if deadline_ms is not None and wait_ms > deadline_ms:
                self._stale_dropped_by_task[task_name] = (
                    self._stale_dropped_by_task.get(task_name, 0) + 1
                )
                continue
            self._queue_wait_ms_by_task.setdefault(task_name, deque(maxlen=2048)).append(wait_ms)
            requests.append(request)
        return requests
