from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlparse


class DashboardApi:
    def __init__(self, debug_service, web_settings) -> None:
        self._debug_service = debug_service
        self._web_settings = web_settings

    def route(self, path: str, query: dict[str, list[str]]) -> tuple[int, str, bytes]:
        if path == "/api/health":
            return self._json(self._debug_service.health_snapshot())
        if path == "/api/status" and self._web_settings.enable_status_api:
            return self._json(self._debug_service.status_snapshot())
        if path == "/api/debug" and self._web_settings.enable_debug_api:
            return self._json(self._debug_service.debug_snapshot(limit=self._extract_limit(query)))
        if path == "/api/logs" and self._web_settings.enable_logs_api:
            return self._json(self._debug_service.logs_snapshot(limit=self._extract_limit(query)))
        if path in {"/", "/index.html"}:
            return self._static("index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._static("app.js", "application/javascript; charset=utf-8")
        return self._json({"error": "not_found", "path": path}, status=HTTPStatus.NOT_FOUND)

    def _extract_limit(self, query: dict[str, list[str]]) -> int:
        raw = query.get("limit", ["100"])[0]
        try:
            return int(raw)
        except ValueError:
            return 100

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> tuple[int, str, bytes]:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return status, "application/json; charset=utf-8", body

    def _static(self, name: str, content_type: str) -> tuple[int, str, bytes]:
        path = Path(__file__).resolve().parent / "static" / name
        if not path.exists():
            return self._json({"error": "static_asset_missing", "asset": name}, status=HTTPStatus.NOT_FOUND)
        return HTTPStatus.OK, content_type, path.read_bytes()


class DashboardServer:
    def __init__(self, debug_service, web_settings) -> None:
        self._api = DashboardApi(debug_service, web_settings)
        self._host = web_settings.host
        self._port = web_settings.port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._httpd is not None:
            return

        api = self._api

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                status, content_type, body = api.route(parsed.path, parse_qs(parsed.query))
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                logging.debug("dashboard %s", format % args)

        self._httpd = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = Thread(target=self._httpd.serve_forever, name="dashboard-server", daemon=True)
        self._thread.start()
        logging.info("dashboard started at http://%s:%s", self._host, self._port)

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None
        logging.info("dashboard stopped")

