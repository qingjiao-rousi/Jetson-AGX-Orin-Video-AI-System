from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any

from app.domain.entities import BoundingBox, FrameResult, Track, canonical_stream_id


@dataclass(frozen=True)
class TaskRequest:
    """A routing decision produced from one tracked target.

    This is deliberately only a request.  Model execution and ROI extraction
    will be added in the next step.
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


@dataclass
class _TrackTaskState:
    last_seen_frame: int = -1
    stable_frames: int = 0
    last_submitted_frame: int = -1


class RoutingPolicy:
    """Map probe results to configured, scene-aware model task requests."""

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
        profile = self._profile_for(result.stream_id)
        if profile is None:
            with self._lock:
                self._unknown_streams += 1
            return ()

        requests: list[TaskRequest] = []
        capability_names = tuple(getattr(profile, "capabilities", ()))
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
        state_key = (canonical_stream_id(result.stream_id), task_name, -1)
        with self._lock:
            state = self._states.setdefault(state_key, _TrackTaskState())
            interval = max(int(task.interval), 1)
            if state.last_submitted_frame >= 0 and result.frame_id - state.last_submitted_frame < interval:
                self._filtered += 1
                return False
            state.last_submitted_frame = result.frame_id
            self._submitted += 1
            return True

    def stats(self) -> dict[str, int]:
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
        if track.track_id < 0 or track.track_id == 0xFFFFFFFFFFFFFFFF:
            with self._lock:
                self._filtered += 1
            return False
        state_key = (canonical_stream_id(result.stream_id), task_name, track.track_id)
        with self._lock:
            state = self._states.setdefault(state_key, _TrackTaskState())
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
    """Bounded latest-task buffer for the next inference execution step."""

    def __init__(self, max_size: int = 32) -> None:
        self._max_size = max(int(max_size), 1)
        self._lock = Lock()
        self._order: deque[tuple[str, str, int]] = deque()
        self._items: dict[tuple[str, str, int], TaskRequest] = {}
        self._dropped = 0

    def submit(self, requests: tuple[TaskRequest, ...] | list[TaskRequest]) -> None:
        with self._lock:
            for request in requests:
                key = (request.stream_id, request.task_name, request.track_id)
                if key in self._items:
                    self._items[key] = request
                    continue
                while len(self._items) >= self._max_size:
                    old_key = self._order.popleft()
                    if old_key in self._items:
                        del self._items[old_key]
                        self._dropped += 1
                self._items[key] = request
                self._order.append(key)

    def drain(self, limit: int | None = None, *, task_name: str | None = None) -> tuple[TaskRequest, ...]:
        with self._lock:
            count = len(self._order) if limit is None else max(int(limit), 0)
            requests: list[TaskRequest] = []
            scanned = 0
            while self._order and len(requests) < count and scanned < len(self._order) + 1:
                key = self._order.popleft()
                request = self._items.pop(key, None)
                scanned += 1
                if request is None:
                    continue
                if task_name is not None and request.task_name != task_name:
                    self._order.append(key)
                    self._items[key] = request
                    continue
                requests.append(request)
            return tuple(requests)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"pending": len(self._items), "dropped": self._dropped}

    def has_pending(self, task_name: str | None = None) -> bool:
        with self._lock:
            if task_name is None:
                return bool(self._items)
            return any(item.task_name == task_name for item in self._items.values())
