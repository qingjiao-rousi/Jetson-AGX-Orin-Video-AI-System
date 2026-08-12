from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ProfilingScriptTests(unittest.TestCase):
    def test_shell_scripts_have_valid_bash_syntax(self) -> None:
        for relative_path in (
            "scripts/benchmark/profile_primary_tensorrt.sh",
            "scripts/benchmark/profile_pipeline_nsys.sh",
        ):
            completed = subprocess.run(
                ["bash", "-n", str(ROOT / relative_path)], capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
