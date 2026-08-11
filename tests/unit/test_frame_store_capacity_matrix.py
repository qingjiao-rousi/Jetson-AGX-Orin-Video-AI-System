from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "run_frame_store_capacity_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_frame_store_capacity_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrameStoreCapacityMatrixTests(unittest.TestCase):
    def test_configures_equal_total_capacity_for_shared_and_per_stream(self) -> None:
        base = {"optimization": {}, "app": {}, "sources": [{}, {}]}
        shared = MODULE.configure_frame_store(
            base, mode="shared", per_stream_capacity=16, source_count=2
        )
        per_stream = MODULE.configure_frame_store(
            base, mode="per_stream", per_stream_capacity=16, source_count=2
        )
        self.assertEqual(shared["optimization"]["frame_store_max_size"], 32)
        self.assertNotIn("frame_store_per_stream_capacity", shared["optimization"])
        self.assertEqual(per_stream["optimization"]["frame_store_max_size"], 32)
        self.assertEqual(per_stream["optimization"]["frame_store_per_stream_capacity"], 16)

    def test_counts_enabled_sources(self) -> None:
        self.assertEqual(
            MODULE.enabled_source_count({"sources": [{}, {"enabled": False}, {"enabled": True}]}),
            2,
        )


if __name__ == "__main__":
    unittest.main()
