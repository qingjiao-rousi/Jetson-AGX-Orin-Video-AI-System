from __future__ import annotations

import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
