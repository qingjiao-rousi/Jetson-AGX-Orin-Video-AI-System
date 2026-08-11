from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
from threading import Lock
from typing import Any


class InMemoryLogBuffer:
    def __init__(self, capacity: int = 200) -> None:
        self._items: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = Lock()

    def append(self, record: logging.LogRecord) -> None:
        entry = {
            "timestamp": self._format_time(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        with self._lock:
            self._items.append(entry)

    def tail(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, limit)
        with self._lock:
            return list(self._items)[-safe_limit:]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            size = len(self._items)
            capacity = self._items.maxlen or 0
        return {
            "size": size,
            "capacity": capacity,
            "is_full": bool(capacity and size >= capacity),
        }

    @staticmethod
    def _format_time(record: logging.LogRecord) -> str:
        from datetime import datetime

        return datetime.fromtimestamp(record.created).isoformat()


class InMemoryLogHandler(logging.Handler):
    def __init__(self, buffer: InMemoryLogBuffer) -> None:
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(record)


def setup_logging(settings, buffer_size: int = 200) -> InMemoryLogBuffer:
    level = getattr(logging, str(getattr(settings, "level", "INFO")).upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_buffer = InMemoryLogBuffer(capacity=buffer_size)

    handlers: list[logging.Handler] = [InMemoryLogHandler(log_buffer)]

    if bool(getattr(settings, "console", True)):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    file_path = Path(getattr(settings, "file_path", "outputs/logs/app.log"))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    for handler in handlers:
        if handler.formatter is None:
            handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return log_buffer
