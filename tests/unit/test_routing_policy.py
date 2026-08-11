from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import unittest

from app.application.routing_policy import RoutingPolicy, TaskRequestBuffer
from app.domain.entities import BoundingBox, FrameResult, Track
from app.settings import (
    AppSettings,
    CapabilitySettings,
    ModelSettings,
    ModelTaskSettings,
    SceneSettings,
    SourceSettings,
)


def make_frame(frame_id: int, *, stream_id: str = "stream-0") -> FrameResult:
    return FrameResult(
        stream_id=stream_id,
        frame_id=frame_id,
        timestamp=datetime.now(timezone.utc),
        tracks=[
            Track(
                track_id=7,
                class_id=0,
                class_name="person",
                confidence=0.95,
                bbox=BoundingBox(left=10, top=20, width=100, height=200),
            )
        ],
    )


class RoutingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings(
            scenes=(SceneSettings(name="normal"), SceneSettings(name="production")),
            sources=(
                SourceSettings(
                    name="production-01",
                    uri="sample.mp4",
                    scene="production",
                    priority="high",
                    capabilities=("helmet_compliance",),
                ),
            ),
            models=(ModelSettings(name="helmet", engine_path=Path("models/helmet.engine")),),
            model_tasks=(
                ModelTaskSettings(
                    name="helmet",
                    model="helmet",
                    trigger_classes=("person",),
                    interval=2,
                    min_track_frames=3,
                ),
            ),
            capabilities=(CapabilitySettings(name="helmet_compliance", tasks=("helmet",)),),
        )
        self.settings.validate()

    def test_routes_only_after_track_is_stable_and_respects_interval(self) -> None:
        policy = RoutingPolicy(self.settings)

        self.assertEqual(policy.route(make_frame(0)), ())
        self.assertEqual(policy.route(make_frame(1)), ())
        first = policy.route(make_frame(2))
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].task_name, "helmet")
        self.assertEqual(first[0].source_name, "production-01")
        self.assertEqual(policy.route(make_frame(3)), ())
        self.assertEqual(len(policy.route(make_frame(4))), 1)

    def test_different_scene_without_capability_does_not_route(self) -> None:
        source = self.settings.sources[0]
        self.settings = AppSettings(
            scenes=self.settings.scenes,
            sources=(SourceSettings(name=source.name, uri=source.uri, scene="normal"),),
            models=self.settings.models,
            model_tasks=self.settings.model_tasks,
            capabilities=self.settings.capabilities,
        )
        policy = RoutingPolicy(self.settings)
        self.assertEqual(policy.route(make_frame(0)), ())

    def test_buffer_is_bounded_and_drains_latest_task(self) -> None:
        policy = RoutingPolicy(self.settings)
        buffer = TaskRequestBuffer(max_size=1)
        request = policy.route(make_frame(0))
        self.assertEqual(request, ())
        request = policy.route(make_frame(1))
        self.assertEqual(request, ())
        request = policy.route(make_frame(2))
        buffer.submit(request)
        self.assertEqual(buffer.stats()["pending"], 1)
        drained = buffer.drain()
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0].track_id, 7)

    def test_buffer_reports_per_task_replacements_and_drops(self) -> None:
        policy = RoutingPolicy(self.settings)
        buffer = TaskRequestBuffer(max_size=1)
        helmet = policy.route(make_frame(0))
        helmet = policy.route(make_frame(1))
        helmet = policy.route(make_frame(2))
        buffer.submit(helmet)
        buffer.submit(helmet)
        stats = buffer.stats()
        self.assertEqual(stats["by_task"]["helmet"]["submitted"], 2)
        self.assertEqual(stats["by_task"]["helmet"]["replaced"], 1)

    def test_buffer_reports_queue_wait_for_drained_requests(self) -> None:
        policy = RoutingPolicy(self.settings)
        buffer = TaskRequestBuffer(max_size=1)
        request = policy.route(make_frame(0))
        request = policy.route(make_frame(1))
        request = policy.route(make_frame(2))
        submitted_at = request[0].submitted_at_monotonic
        buffer.submit(request)
        with patch("app.application.routing_policy.time.monotonic", return_value=submitted_at + 0.125):
            buffer.drain()
        queue_wait = buffer.stats()["by_task"]["helmet"]["queue_wait_ms"]
        self.assertEqual(queue_wait["samples"], 1)
        self.assertEqual(queue_wait["p50"], 125.0)

    def test_buffer_isolates_task_capacities(self) -> None:
        task_settings = (
            ModelTaskSettings(name="helmet", model="helmet", queue_size=2),
            ModelTaskSettings(name="pose", model="helmet", queue_size=1),
        )
        buffer = TaskRequestBuffer(max_size=1, task_settings=task_settings)
        policy = RoutingPolicy(self.settings)
        policy.route(make_frame(0))
        policy.route(make_frame(1))
        helmet = policy.route(make_frame(2))[0]
        pose = replace(helmet, task_name="pose")
        buffer.submit((helmet, pose))
        stats = buffer.stats()
        self.assertEqual(stats["pending"], 2)
        self.assertEqual(stats["by_task"]["helmet"]["queue_size"], 2)
        self.assertEqual(stats["by_task"]["pose"]["queue_size"], 1)

    def test_buffer_discards_stale_request_at_drain(self) -> None:
        task_settings = (ModelTaskSettings(name="helmet", model="helmet", stale_after_ms=50),)
        buffer = TaskRequestBuffer(task_settings=task_settings)
        policy = RoutingPolicy(self.settings)
        policy.route(make_frame(0))
        policy.route(make_frame(1))
        request = policy.route(make_frame(2))[0]
        buffer.submit((request,))
        with patch(
            "app.application.routing_policy.time.monotonic",
            return_value=request.submitted_at_monotonic + 0.051,
        ):
            self.assertEqual(buffer.drain(task_name="helmet"), ())
        stats = buffer.stats()["by_task"]["helmet"]
        self.assertEqual(stats["stale_dropped"], 1)
        self.assertEqual(stats["queue_wait_ms"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
