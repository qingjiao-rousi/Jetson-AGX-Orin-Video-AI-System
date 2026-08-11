from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_benchmark_matrix import override_primary_engine


class BenchmarkMatrixTests(unittest.TestCase):
    def test_primary_engine_override_updates_both_required_paths(self) -> None:
        source = {
            "models": {"primary": {"engine": "models/int8/old.engine"}},
            "deepstream": {"model_engine_path": "models/int8/old.engine"},
        }
        result = override_primary_engine(source, Path("models/int8/candidate.engine"))
        self.assertEqual(result["models"]["primary"]["engine"], "models/int8/candidate.engine")
        self.assertEqual(result["deepstream"]["model_engine_path"], "models/int8/candidate.engine")
        self.assertEqual(source["models"]["primary"]["engine"], "models/int8/old.engine")


if __name__ == "__main__":
    unittest.main()
