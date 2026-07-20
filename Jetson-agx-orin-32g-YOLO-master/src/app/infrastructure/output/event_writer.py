from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from threading import Lock
from typing import Any


class EventWriter:
    """Small thread-safe JSONL writer for asynchronous business events."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        self._lock = Lock()
        self._written = 0
        self._errors = 0

    def write(self, event: object) -> None:
        payload = asdict(event) if is_dataclass(event) else dict(event) if isinstance(event, dict) else event
        try:
            line = json.dumps(payload, ensure_ascii=False, default=self._json_default)
            with self._lock:
                self._file.write(line + "\n")
                self._file.flush()
                self._written += 1
        except Exception:
            with self._lock:
                self._errors += 1
            raise

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self._path),
                "written": self._written,
                "errors": self._errors,
            }

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"unsupported event value: {type(value).__name__}")
