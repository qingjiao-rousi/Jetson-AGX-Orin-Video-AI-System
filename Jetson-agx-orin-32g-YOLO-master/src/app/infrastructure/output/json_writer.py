from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any, Callable, TextIO
import logging


_SENTINEL = object()


class JsonWriter:
    """Write structured results off the DeepStream probe thread."""

    def __init__(
        self,
        path: Path,
        *,
        queue_size: int = 32,
        drop_oldest: bool = True,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be greater than zero")
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("a", encoding="utf-8")
        self._queue: Queue[object] = Queue(maxsize=queue_size)
        self._drop_oldest = drop_oldest
        self._on_error = on_error
        self._lock = Lock()
        self._closed = False
        self._worker_error: str | None = None
        self._lines_written = 0
        self._enqueued = 0
        self._dropped = 0
        self._write_errors = 0
        self._worker = Thread(target=self._run, name="json-result-writer", daemon=True)
        self._worker.start()

    def write(self, result: object) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("JsonWriter is closed")

        try:
            self._queue.put_nowait(result)
            with self._lock:
                self._enqueued += 1
            return
        except Full:
            pass

        if not self._drop_oldest:
            with self._lock:
                self._dropped += 1
            return

        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except Empty:
            pass

        try:
            self._queue.put_nowait(result)
            with self._lock:
                self._enqueued += 1
                self._dropped += 1
        except Full:
            with self._lock:
                self._dropped += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        # A blocking sentinel guarantees that all queued results are drained
        # before the worker exits, even when the queue is full.
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            logging.error("JSON writer worker did not stop before timeout")
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "path": str(self._path),
                "lines_written": self._lines_written,
                "enqueued": self._enqueued,
                "dropped": self._dropped,
                "write_errors": self._write_errors,
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "worker_alive": self._worker.is_alive(),
                "worker_error": self._worker_error,
                "is_closed": self._file.closed,
            }

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                self._write_item(item)
            except Exception as exc:
                message = f"JSON result writer failed: {exc}"
                with self._lock:
                    self._write_errors += 1
                    self._worker_error = message
                logging.exception(message)
                if self._on_error is not None:
                    try:
                        self._on_error(message)
                    except Exception:
                        logging.exception("JSON writer error callback failed")
            finally:
                self._queue.task_done()

    def _write_item(self, result: object) -> None:
        payload = self._to_jsonable(asdict(result))
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            if self._file.closed:
                raise RuntimeError("JSON output file is closed")
            self._file.write(line + "\n")
            self._file.flush()
            self._lines_written += 1

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, datetime):
            timestamp = value
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc).isoformat()
        if isinstance(value, dict):
            return {key: self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [self._to_jsonable(item) for item in value]
        return value
