from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import TextIO


class JsonWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("a", encoding="utf-8")
        self._lock = Lock()
        self._lines_written = 0

    def write(self, result: object) -> None:
        payload = asdict(result)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
            self._lines_written += 1

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def stats(self) -> dict[str, object]:
        return {
            "path": str(self._path),
            "lines_written": self._lines_written,
            "is_closed": self._file.closed,
        }
