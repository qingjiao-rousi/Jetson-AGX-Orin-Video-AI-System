from __future__ import annotations

"""将主 ``FrameResult`` 异步写入 JSONL，避免磁盘 I/O 阻塞 DeepStream probe 回调。"""

import json
# FrameResult 等领域 dataclass 在 writer 的单独线程中递归转为普通 JSON 值。
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any, Callable, TextIO
import logging


_SENTINEL = object()


class JsonWriter:
    """有界、单消费者的主结果 JSONL writer。

    ``write`` 运行于应用/probe 回调侧，只做非阻塞入队；实际序列化和 flush 在独立
    线程执行。队列满时可选淘汰最旧结果，实时部署优先保留新帧而非保证逐帧存档。
    """

    def __init__(
        self,
        path: Path,
        *,
        queue_size: int = 32,
        drop_oldest: bool = True,
        on_error: Callable[[str], None] | None = None,
        on_written: Callable[[object], None] | None = None,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be greater than zero")
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("a", encoding="utf-8")
        # 主线程只做 put_nowait；单消费者保证 JSONL 行序与成功写出回调一致。
        self._queue: Queue[object] = Queue(maxsize=queue_size)
        self._drop_oldest = drop_oldest
        self._on_error = on_error
        self._on_written = on_written
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
        """非阻塞提交结果；满队列时按配置丢弃当前项或替换队首旧项。"""
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
            # 丢弃当前新结果适合完整性优先模式；实时实验默认改为淘汰更旧的队首。
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
        """停止接收新结果后发送 sentinel，并等待已入队项目写完再关闭文件。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        # 阻塞写入 sentinel 保证 worker 在退出前 drain 已入队项目，即使队列正满。
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            logging.error("JSON writer worker did not stop before timeout")
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def stats(self) -> dict[str, object]:
        """提供 writer 队列深度与丢弃量，是背压和运行时指标的输入。"""
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
        """单消费者顺序落盘；写入失败上报 pipeline，但线程继续消费后续项目。"""
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                self._write_item(item)
                # 只有实际 flush 成功后才通知背压/端到端时延，入队不等于已消费。
                if self._on_written is not None:
                    self._on_written(item)
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
        """将 dataclass 递归变为 JSON 可表示类型并立即 flush，便于实验中断后保留证据。"""
        payload = self._to_jsonable(asdict(result))
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            if self._file.closed:
                raise RuntimeError("JSON output file is closed")
            self._file.write(line + "\n")
            self._file.flush()
            self._lines_written += 1

    def _to_jsonable(self, value: Any) -> Any:
        """统一时间为 UTC ISO-8601，递归处理 dataclasses 转换得到的容器。"""
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
