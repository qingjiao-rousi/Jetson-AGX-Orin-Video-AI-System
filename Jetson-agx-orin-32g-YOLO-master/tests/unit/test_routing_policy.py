from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
