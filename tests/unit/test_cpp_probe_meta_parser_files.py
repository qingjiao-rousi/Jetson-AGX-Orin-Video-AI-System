from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "custom_libs" / "probe_handler"


class CppProbeMetaParserFileTests(unittest.TestCase):
    def test_scaffold_files_exist(self) -> None:
        expected = {
            "CMakeLists.txt",
            "README.md",
            "include/probe_handler/probe_meta_parser.h",
            "src/probe_meta_parser.cpp",
        }

        actual = {
            str(path.relative_to(PROBE_DIR)).replace("\\", "/")
            for path in PROBE_DIR.rglob("*")
            if path.is_file()
        }

        self.assertTrue(expected <= actual)

    def test_exposes_stable_c_api_and_deepstream_path(self) -> None:
        header = (PROBE_DIR / "include" / "probe_handler" / "probe_meta_parser.h").read_text(
            encoding="utf-8"
        )
        source = (PROBE_DIR / "src" / "probe_meta_parser.cpp").read_text(encoding="utf-8")
        cmake = (PROBE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("struct ProbeFrameResult", header)
        self.assertIn("struct ProbeDetection", header)
        self.assertIn("parse_nvds_batch_meta", header)
        self.assertIn('extern "C"', header)
        self.assertIn("probe_parse_nvds_batch_meta_json", header)
        self.assertIn("probe_parse_gst_buffer_json", header)
        self.assertIn("nvdsmeta.h", source)
        self.assertIn("NVDS_VERSION_MAJOR", cmake)
        self.assertIn("probe_handler", cmake)


if __name__ == "__main__":
    unittest.main()
