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
                        "batch-size=1",
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
            self.assertIn("batch-size=1", runtime_text)
            self.assertIn("interval=1", runtime_text)

    def test_runtime_override_can_restore_per_frame_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            infer_config = Path(tmp) / "infer.txt"
            infer_config.write_text("[property]\ninterval=1\n", encoding="utf-8")
            settings = AppSettings(
                deepstream=DeepStreamSettings(infer_config_path=infer_config, infer_interval=0)
            )

            updated = apply_runtime_overrides(settings)

            self.assertIn("interval=0", updated.deepstream.infer_config_path.read_text(encoding="utf-8"))

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

    def test_runtime_dir_isolated_per_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            infer_config = root / "infer.txt"
            runtime_dir = root / "run-001" / ".runtime"
            infer_config.write_text(
                "[property]\n\n[class-attrs-all]\npre-cluster-threshold=0.25\n",
                encoding="utf-8",
            )
            settings = AppSettings(deepstream=DeepStreamSettings(infer_config_path=infer_config))

            updated = apply_runtime_overrides(
                settings,
                confidence_threshold=0.4,
                runtime_dir=runtime_dir,
            )

            self.assertEqual(updated.deepstream.infer_config_path, runtime_dir / "infer.txt")
            self.assertTrue(updated.deepstream.infer_config_path.exists())
            self.assertIn(
                "pre-cluster-threshold=0.4000",
                updated.deepstream.infer_config_path.read_text(encoding="utf-8"),
            )

    def test_output_directory_override_configures_all_runtime_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            infer_config = root / "infer.txt"
            output_dir = root / "run-002"
            infer_config.write_text("[property]\n", encoding="utf-8")
            settings = AppSettings(deepstream=DeepStreamSettings(infer_config_path=infer_config))

            updated = apply_runtime_overrides(
                settings,
                output_dir=output_dir,
                output_sink="rtmp",
                output_url="rtmp://127.0.0.1/live/test",
            )

            self.assertEqual(updated.output.jsonl_path, output_dir / "results.jsonl")
            self.assertEqual(updated.output.metrics_jsonl_path, output_dir / "runtime_metrics.jsonl")
            self.assertEqual(updated.logging.file_path, output_dir / "app.log")
            self.assertEqual(updated.deepstream.output_video_path, output_dir / "output.mp4")
            self.assertEqual(updated.deepstream.output_sink, "rtmp")
            self.assertEqual(updated.deepstream.output_url, "rtmp://127.0.0.1/live/test")


if __name__ == "__main__":
    unittest.main()
