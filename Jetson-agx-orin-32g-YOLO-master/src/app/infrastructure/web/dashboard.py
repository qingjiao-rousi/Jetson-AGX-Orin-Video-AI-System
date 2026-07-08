from __future__ import annotations

import json
import logging
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, unquote, urlparse


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
        if path == "/api/batch/summary":
            return self._batch_json("batch_summary.json")
        if path == "/api/batch/quality":
            return self._batch_json("batch_quality.json")
        if path == "/api/batch/dashboard":
            return self._batch_dashboard()
        if path.startswith("/batch-files/"):
            return self._batch_file(path.removeprefix("/batch-files/"))
        if path in {"/", "/index.html"}:
            return self._static("index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._static("app.js", "application/javascript; charset=utf-8")
        return self._json({"error": "not_found", "path": path}, status=HTTPStatus.NOT_FOUND)

    def batch_file_response(self, raw_relative_path: str, range_header: str | None = None) -> tuple[int, str, bytes, dict[str, str]]:
        relative_path = Path(unquote(raw_relative_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            status, content_type, body = self._json({"error": "invalid_batch_file_path"}, status=HTTPStatus.BAD_REQUEST)
            return status, content_type, body, {}
        path = (self._batch_dir() / relative_path).resolve()
        batch_dir = self._batch_dir().resolve()
        if batch_dir not in path.parents and path != batch_dir:
            status, content_type, body = self._json({"error": "batch_file_outside_root"}, status=HTTPStatus.BAD_REQUEST)
            return status, content_type, body, {}
        if not path.is_file():
            status, content_type, body = self._json(
                {"error": "batch_file_missing", "file": str(relative_path)},
                status=HTTPStatus.NOT_FOUND,
            )
            return status, content_type, body, {}

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        headers = {"Accept-Ranges": "bytes"}
        byte_range = self._parse_range_header(range_header, size)
        if byte_range is None:
            return HTTPStatus.OK, content_type, path.read_bytes(), headers

        start, end = byte_range
        length = end - start + 1
        with path.open("rb") as file:
            file.seek(start)
            body = file.read(length)
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return HTTPStatus.PARTIAL_CONTENT, content_type, body, headers

    def _extract_limit(self, query: dict[str, list[str]]) -> int:
        raw = query.get("limit", ["100"])[0]
        try:
            return int(raw)
        except ValueError:
            return 100

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> tuple[int, str, bytes]:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return status, "application/json; charset=utf-8", body

    def _batch_dashboard(self) -> tuple[int, str, bytes]:
        summary = self._read_batch_json("batch_summary.json")
        quality = self._read_batch_json("batch_quality.json")
        payload = {
            "batch_dir": str(self._batch_dir()),
            "summary": summary,
            "quality": quality,
            "artifacts": self._batch_artifacts(),
            "videos": self._merge_batch_videos(summary, quality),
        }
        return self._json(payload)

    def _batch_json(self, name: str) -> tuple[int, str, bytes]:
        path = self._batch_dir() / name
        if not path.exists():
            return self._json(
                {"error": "batch_file_missing", "file": name, "batch_dir": str(self._batch_dir())},
                status=HTTPStatus.NOT_FOUND,
            )
        return HTTPStatus.OK, "application/json; charset=utf-8", path.read_bytes()

    def _batch_file(self, raw_relative_path: str) -> tuple[int, str, bytes]:
        status, content_type, body, _headers = self.batch_file_response(raw_relative_path)
        return status, content_type, body

    def _parse_range_header(self, range_header: str | None, size: int) -> tuple[int, int] | None:
        if not range_header or not range_header.startswith("bytes=") or size <= 0:
            return None
        raw_range = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
        if "-" not in raw_range:
            return None
        raw_start, raw_end = raw_range.split("-", 1)
        try:
            if raw_start == "":
                suffix_length = int(raw_end)
                if suffix_length <= 0:
                    return None
                start = max(size - suffix_length, 0)
                end = size - 1
            else:
                start = int(raw_start)
                end = int(raw_end) if raw_end else size - 1
        except ValueError:
            return None
        if start < 0 or end < start or start >= size:
            return None
        return start, min(end, size - 1)

    def _batch_dir(self) -> Path:
        return Path(getattr(self._web_settings, "batch_dir", Path("outputs/batch")))

    def _read_batch_json(self, name: str) -> dict:
        path = self._batch_dir() / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _merge_batch_videos(self, summary: dict, quality: dict) -> list[dict]:
        quality_by_input = {
            item.get("input_video"): item
            for item in quality.get("videos", [])
            if isinstance(item, dict) and item.get("input_video")
        }
        videos = []
        for index, video in enumerate(summary.get("videos", []), start=1):
            if not isinstance(video, dict):
                continue
            merged = dict(video)
            merged["index"] = index
            quality_item = quality_by_input.get(video.get("input_video"), {})
            merged["quality"] = quality_item
            for key in ("output_video", "output_overlay_video", "output_jsonl", "output_summary", "log_path"):
                if merged.get(key):
                    merged[f"{key}_url"] = self._batch_url_for_path(merged[key])
            if merged.get("log_path"):
                merged["log_tail"] = self._read_text_tail(Path(str(merged["log_path"])), max_lines=80)
            videos.append(merged)
        return videos

    def _batch_artifacts(self) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        for key, name in {
            "summary": "batch_summary.json",
            "quality": "batch_quality.json",
            "html_report": "batch_report.html",
            "csv_report": "batch_summary.csv",
        }.items():
            path = self._batch_dir() / name
            if path.is_file():
                artifacts[key] = self._batch_url_for_path(str(path))
        return artifacts

    def _batch_url_for_path(self, raw_path: str) -> str:
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            relative = path.resolve().relative_to(self._batch_dir().resolve())
        except ValueError:
            return ""
        return "/batch-files/" + relative.as_posix()

    def _read_text_tail(self, path: Path, max_lines: int = 80) -> str:
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-max_lines:])

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
                self._send_response(send_body=True)

            def do_HEAD(self) -> None:  # noqa: N802
                self._send_response(send_body=False)

            def _send_response(self, send_body: bool) -> None:
                parsed = urlparse(self.path)
                extra_headers = {}
                if parsed.path.startswith("/batch-files/"):
                    status, content_type, body, extra_headers = api.batch_file_response(
                        parsed.path.removeprefix("/batch-files/"),
                        self.headers.get("Range"),
                    )
                else:
                    status, content_type, body = api.route(parsed.path, parse_qs(parsed.query))
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for name, value in extra_headers.items():
                    self.send_header(name, value)
                self.end_headers()
                if send_body:
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
