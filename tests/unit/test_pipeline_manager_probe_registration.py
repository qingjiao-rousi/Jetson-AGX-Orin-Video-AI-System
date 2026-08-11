from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from app.infrastructure.pipeline.manager import PipelineManager


class FakeBuilder:
    def __init__(self) -> None:
        self.calls = 0

    def attach_probe_points(self, runtime):
        self.calls += 1
        return ({"element": "osd", "pad": "sink"},)


class PipelineManagerProbeRegistrationTests(unittest.TestCase):
    def test_probe_registration_is_owned_by_manager_and_is_idempotent(self) -> None:
        builder = FakeBuilder()
        manager = PipelineManager(builder)
        manager._runtime = {"blueprint": object(), "probe_attachments": ()}
        manager._runtime["probe_registry"] = manager.probes()
        manager._runtime["meta_parser"] = object()

        first = manager._register_probe_points()
        second = manager._register_probe_points()

        self.assertEqual(first, second)
        self.assertEqual(builder.calls, 1)
        self.assertTrue(manager._runtime["probe_points_registered"])


if __name__ == "__main__":
    unittest.main()
