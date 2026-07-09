from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_multifile_inproc_outputs import check_multifile
from summarize_multifile_inproc import summarize_multifile


class MultifileInprocSummaryTests(unittest.TestCase):
    def test_summarize_and_check_multistream_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = [
                _row("stream-0", 0, "2026-07-09 00:00:00+00:00", [1]),
                _row("stream-0", 1, "2026-07-09 00:00:00.020000+00:00", [1]),
                _row("stream-1", 0, "2026-07-09 00:00:01+00:00", [2]),
                _row("stream-1", 1, "2026-07-09 00:00:01.020000+00:00", [2]),
            ]
            (output_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            (output_dir / "multifile_preview.mp4").write_bytes(b"mp4")
            (output_dir / "run.log").write_text("done\n", encoding="utf-8")
            (output_dir / "run_metadata.json").write_text(
                json.dumps({"status": "ok", "exit_code": 0, "input_videos": ["/videos/a.mp4", "/videos/b.mp4"]}),
                encoding="utf-8",
            )
            runtime_dir = output_dir / ".runtime"
            runtime_dir.mkdir()
            (runtime_dir / "app_multifile_runtime.yaml").write_text("app: {}\n", encoding="utf-8")

            summary = summarize_multifile(output_dir, expected_stream_count=2)
            summary_path = output_dir / "multifile_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            quality = check_multifile(summary_path)

        self.assertEqual(summary["observed_stream_count"], 2)
        self.assertEqual(summary["missing_stream_ids"], [])
        self.assertEqual(summary["total_frame_count"], 4)
        self.assertEqual(summary["total_unique_persons"], 2)
        self.assertEqual(summary["streams"]["stream-0"]["total_unique_persons"], 1)
        self.assertEqual(quality["quality_status"], "passed")
        self.assertEqual(quality["passed_stream_count"], 2)

    def test_missing_stream_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "results.jsonl").write_text(
                json.dumps(_row("stream-0", 0, "2026-07-09 00:00:00+00:00", [1])) + "\n",
                encoding="utf-8",
            )
            (output_dir / "multifile_preview.mp4").write_bytes(b"mp4")
            (output_dir / "run.log").write_text("done\n", encoding="utf-8")
            (output_dir / "run_metadata.json").write_text(
                json.dumps({"status": "ok", "exit_code": 0, "input_videos": ["/videos/a.mp4", "/videos/b.mp4"]}),
                encoding="utf-8",
            )
            summary = summarize_multifile(output_dir, expected_stream_count=2)
            summary_path = output_dir / "multifile_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            quality = check_multifile(summary_path)

        self.assertEqual(summary["missing_stream_ids"], ["stream-1"])
        self.assertEqual(quality["quality_status"], "failed")
        self.assertTrue(any("missing streams" in item for item in quality["failures"]))

    def test_failed_run_metadata_is_reported_in_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "results.jsonl").write_text("", encoding="utf-8")
            (output_dir / "run.log").write_text("Traceback: boom\n", encoding="utf-8")
            (output_dir / "run_metadata.json").write_text(
                json.dumps({"status": "failed", "exit_code": 2, "error": "app.main exited with 2"}),
                encoding="utf-8",
            )
            summary = summarize_multifile(output_dir, expected_stream_count=1)
            summary_path = output_dir / "multifile_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            quality = check_multifile(summary_path)

        self.assertEqual(summary["run_status"], "failed")
        self.assertEqual(quality["quality_status"], "failed")
        self.assertIn("run status is failed", quality["failures"])
        self.assertIn("app.main exited with 2", quality["failures"])
        self.assertTrue(any("run log fatal" in item for item in quality["failures"]))

    def test_malformed_jsonl_rows_are_counted_and_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "results.jsonl").write_text(
                json.dumps(_row("stream-0", 0, "2026-07-09 00:00:00+00:00", [1])) + "\n{bad json\n",
                encoding="utf-8",
            )
            (output_dir / "multifile_preview.mp4").write_bytes(b"mp4")
            (output_dir / "run.log").write_text("done\n", encoding="utf-8")
            (output_dir / "run_metadata.json").write_text(
                json.dumps({"status": "ok", "exit_code": 0, "input_videos": ["/videos/a.mp4"]}),
                encoding="utf-8",
            )
            summary = summarize_multifile(output_dir, expected_stream_count=1)
            summary_path = output_dir / "multifile_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            quality = check_multifile(summary_path)

        self.assertEqual(summary["malformed_json_line_count"], 1)
        self.assertEqual(quality["quality_status"], "failed")
        self.assertIn("results JSONL has 1 malformed rows", quality["failures"])


def _row(stream_id: str, frame_id: int, timestamp: str, track_ids: list[int]) -> dict:
    return {
        "stream_id": stream_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.9,
                "bbox": {"left": 1, "top": 2, "width": 3, "height": 4},
            }
        ],
        "tracks": [
            {
                "track_id": track_id,
                "class_id": 0,
                "confidence": 0.9,
                "bbox": {"left": 1, "top": 2, "width": 3, "height": 4},
            }
            for track_id in track_ids
        ],
        "extra": {},
    }


if __name__ == "__main__":
    unittest.main()
