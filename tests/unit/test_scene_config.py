from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.adapters.config_loader import load_settings
from app.settings import AppSettings, CapabilitySettings, SceneSettings, SourceSettings


class SceneConfigTests(unittest.TestCase):
    def test_loads_scene_and_camera_profile_fields(self) -> None:
        config = """
scenes:
  production:
    description: 生产作业区
  warehouse:
    description: 仓库物流通道
models:
  helmet:
    backend: tensorrt
    engine: models/helmet.engine
model_tasks:
  helmet:
    model: helmet
    trigger_classes: [person]
    interval: 2
    min_track_frames: 3
capabilities:
  helmet_compliance:
    tasks: [helmet]
sources:
  - name: cam-production
    uri: rtsp://127.0.0.1/production
    scene: production
    priority: high
    zones: [production_area, restricted_line]
    capabilities: [helmet_compliance]
  - name: cam-warehouse
    uri: rtsp://127.0.0.1/warehouse
    scene: warehouse
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app.yaml"
            path.write_text(config, encoding="utf-8")
            settings = load_settings(path)

        self.assertEqual(settings.sources[0].scene, "production")
        self.assertEqual(settings.sources[0].priority, "high")
        self.assertEqual(settings.sources[0].zones, ("production_area", "restricted_line"))
        self.assertEqual(settings.sources[0].capabilities, ("helmet_compliance",))
        self.assertEqual(settings.sources[1].scene, "warehouse")
        self.assertEqual(settings.model_tasks[0].model, "helmet")
        self.assertEqual(settings.model_tasks[0].trigger_classes, ("person",))
        self.assertEqual(settings.capabilities[0].tasks, ("helmet",))
        self.assertIn("normal", {scene.name for scene in settings.scenes})
        settings.validate()

    def test_capability_must_reference_a_known_task(self) -> None:
        settings = AppSettings(
            capabilities=(CapabilitySettings(name="helmet_compliance", tasks=("missing",)),),
        )
        with self.assertRaisesRegex(ValueError, "unknown tasks"):
            settings.validate()

    def test_unknown_source_scene_is_rejected(self) -> None:
        settings = AppSettings(
            scenes=(SceneSettings(name="normal"),),
            sources=(
                SourceSettings(
                    name="cam-01",
                    uri="sample.mp4",
                    scene="production",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown scene"):
            settings.validate()

    def test_duplicate_source_names_are_rejected(self) -> None:
        settings = AppSettings(
            sources=(
                SourceSettings(name="cam-01", uri="a.mp4"),
                SourceSettings(name="cam-01", uri="b.mp4"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "source names must be unique"):
            settings.validate()


if __name__ == "__main__":
    unittest.main()
