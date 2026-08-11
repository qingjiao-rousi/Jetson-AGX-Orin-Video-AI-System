from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_ppe_microbatch_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_ppe_microbatch_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PpeMicrobatchMatrixTests(unittest.TestCase):
    def test_batch_two_uses_batch_four_dynamic_profile(self) -> None:
        self.assertEqual(
            MODULE.ppe_engine_for_batch(2),
            "models/fp16/ppe_yolov8n_dynamic_fp16_b4.engine",
        )

    def test_configure_ppe_limits_runtime_batch_to_two(self) -> None:
        config = MODULE.configure_ppe(
            {"models": {"helmet": {}}, "model_tasks": {"helmet": {}}},
            batch_size=2,
            wait_ms=0,
        )
        self.assertEqual(config["models"]["helmet"]["engine"], MODULE.ppe_engine_for_batch(2))
        self.assertEqual(config["model_tasks"]["helmet"]["micro_batch_size"], 2)
        self.assertEqual(config["model_tasks"]["helmet"]["micro_batch_wait_ms"], 0)

    def test_event_consistency_uses_only_coarse_helmet_coverage(self) -> None:
        def run(batch_size: int, signatures: list[list[object]]) -> dict:
            return {
                "batch_size": batch_size,
                "repetition": 1,
                "summary": {"predictions": {"event_signatures": signatures}},
            }

        comparisons = MODULE.event_consistency(
            [
                run(1, [["helmet_violation", "stream-1", 4, 100, "not_wearing"], ["zone_observation", "stream-1"]]),
                run(2, [["helmet_violation", "stream-1", 8, 300, "not_wearing"], ["scene_statistics", "stream-1"]]),
            ]
        )
        self.assertEqual(comparisons[0]["matched_event_keys"], 1)
        self.assertEqual(comparisons[0]["batch1_only_event_keys"], 0)
        self.assertEqual(comparisons[0]["candidate_only_event_keys"], 0)


if __name__ == "__main__":
    unittest.main()
