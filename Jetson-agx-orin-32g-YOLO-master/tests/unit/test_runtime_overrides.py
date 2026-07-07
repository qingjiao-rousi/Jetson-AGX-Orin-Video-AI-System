from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.adapters.runtime_overrides import apply_runtime_overrides
from app.settings import AppSettings, DeepStreamSettings, OutputSettings, SourceSettings


class RuntimeOverrideTests(unittest.TestCase):
    def test_apply_runtime_overrides_updates_paths_and_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            infer_config = root / "infer.txt"
            infer_config.write_text(
                "\n".join(
                    [
                        "[property]",
                        "cluster-mode=2",
                        "filter-out-class-ids=1;2",
                        "",
                        "[class-attrs-all]",
                        "pre-cluster-threshold=0.25",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            settings = AppSettings(
                source_count=1,
                sources=(SourceSettings(name="old", uri="old.mp4", kind="file"),),
                output=OutputSettings(jsonl_path=Path("old.jsonl")),
                deepstream=DeepStreamSettings(
                    infer_config_path=infer_config,
                    output_video_path=Path("old.mp4"),
                    inference_width=640,
                    inference_height=640,
                ),
            )

            updated = apply_runtime_overrides(
                settings,
                input_video=Path("/tmp/input.mp4"),
                output_video=Path("outputs/out.mp4"),
                output_json=Path("outputs/out.jsonl"),
                output_width=1280,
                output_height=720,
                confidence_threshold=0.3,
                enable_web=False,
            )

            self.assertEqual(updated.sources[0].uri, "/tmp/input.mp4")
            self.assertEqual(updated.deepstream.output_video_path, Path("outputs/out.mp4"))
            self.assertEqual(updated.output.jsonl_path, Path("outputs/out.jsonl"))
            self.assertEqual(updated.deepstream.inference_width, 1280)
            self.assertEqual(updated.deepstream.inference_height, 720)
            self.assertFalse(updated.web.enabled)
            runtime_text = updated.deepstream.infer_config_path.read_text(encoding="utf-8")
            self.assertIn("filter-out-class-ids=1;2;3", runtime_text)
            self.assertIn("pre-cluster-threshold=0.3000", runtime_text)

    def test_all_classes_comments_person_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            infer_config = Path(tmp) / "infer.txt"
            infer_config.write_text(
                "[property]\nfilter-out-class-ids=1;2;3\n",
                encoding="utf-8",
            )
            settings = AppSettings(deepstream=DeepStreamSettings(infer_config_path=infer_config))

            updated = apply_runtime_overrides(settings, person_only=False)

            runtime_text = updated.deepstream.infer_config_path.read_text(encoding="utf-8")
            self.assertIn("# filter-out-class-ids=", runtime_text)


if __name__ == "__main__":
    unittest.main()
