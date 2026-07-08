from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.infrastructure.web.dashboard import DashboardApi
from app.settings import WebSettings


class FakeDebugService:
    def health_snapshot(self) -> dict:
        return {"healthy": True}

    def status_snapshot(self) -> dict:
        return {"status": "ok"}

    def debug_snapshot(self, limit: int = 100) -> dict:
        return {"debug": True, "limit": limit}

    def logs_snapshot(self, limit: int = 100) -> dict:
        return {"logs": True, "limit": limit}


class DashboardApiTests(unittest.TestCase):
    def test_status_route_returns_json_payload(self) -> None:
        api = DashboardApi(FakeDebugService(), WebSettings(enabled=True))

        status, content_type, body = api.route("/api/status", {})

        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertIn(b"ok", body)

    def test_logs_route_respects_limit(self) -> None:
        api = DashboardApi(FakeDebugService(), WebSettings(enabled=True))

        status, _content_type, body = api.route("/api/logs", {"limit": ["7"]})

        self.assertEqual(status, 200)
        self.assertIn(b"7", body)

    def test_batch_dashboard_route_merges_summary_quality_and_video_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = Path(tmp)
            output_dir = batch_dir / "001_a"
            output_dir.mkdir()
            video_path = output_dir / "person_analytics_overlay.mp4"
            video_path.write_bytes(b"mp4")
            log_path = output_dir / "run.log"
            log_path.write_text("line1\nline2\n", encoding="utf-8")
            (batch_dir / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "video_count": 1,
                        "videos": [
                            {
                                "input_video": "/videos/a.mp4",
                                "status": "ok",
                                "output_overlay_video": str(video_path),
                                "log_path": str(log_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (batch_dir / "batch_quality.json").write_text(
                json.dumps(
                    {
                        "passed_count": 1,
                        "videos": [
                            {
                                "input_video": "/videos/a.mp4",
                                "quality_status": "passed",
                                "failures": [],
                                "reviews": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (batch_dir / "batch_report.html").write_text("<html></html>", encoding="utf-8")
            (batch_dir / "batch_summary.csv").write_text("video,status\n", encoding="utf-8")
            api = DashboardApi(FakeDebugService(), WebSettings(enabled=True, batch_dir=batch_dir))

            status, content_type, body = api.route("/api/batch/dashboard", {})
            payload = json.loads(body.decode("utf-8"))

            self.assertEqual(status, 200)
            self.assertIn("application/json", content_type)
            self.assertEqual(payload["videos"][0]["quality"]["quality_status"], "passed")
            self.assertEqual(payload["videos"][0]["output_overlay_video_url"], "/batch-files/001_a/person_analytics_overlay.mp4")
            self.assertEqual(payload["videos"][0]["log_path_url"], "/batch-files/001_a/run.log")
            self.assertEqual(payload["artifacts"]["summary"], "/batch-files/batch_summary.json")
            self.assertEqual(payload["artifacts"]["quality"], "/batch-files/batch_quality.json")
            self.assertEqual(payload["artifacts"]["html_report"], "/batch-files/batch_report.html")
            self.assertEqual(payload["artifacts"]["csv_report"], "/batch-files/batch_summary.csv")
            self.assertIn("line2", payload["videos"][0]["log_tail"])

            status, content_type, body = api.route("/batch-files/001_a/person_analytics_overlay.mp4", {})

            self.assertEqual(status, 200)
            self.assertIn("video/mp4", content_type)
            self.assertEqual(body, b"mp4")

            status, content_type, body, headers = api.batch_file_response(
                "001_a/person_analytics_overlay.mp4",
                "bytes=1-2",
            )

            self.assertEqual(status, 206)
            self.assertIn("video/mp4", content_type)
            self.assertEqual(body, b"p4")
            self.assertEqual(headers["Accept-Ranges"], "bytes")
            self.assertEqual(headers["Content-Range"], "bytes 1-2/3")


if __name__ == "__main__":
    unittest.main()
