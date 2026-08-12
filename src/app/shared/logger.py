from __future__ import annotations

"""控制台、文件和 dashboard 调试接口共用的日志装配。"""

from collections import deque
import logging
from pathlib import Path
from threading import Lock
from typing import Any


class InMemoryLogBuffer:
    """线程安全的有界日志环形缓冲，供 Web/debug API 查询最近日志。"""
    def __init__(self, capacity: int = 200) -> None:
        self._items: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = Lock()

    def append(self, record: logging.LogRecord) -> None:
        # 只保存稳定字段，Web 查询无需持有 logging 内部对象。
        entry = {
            "timestamp": self._format_time(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        with self._lock:
            self._items.append(entry)

    def tail(self, limit: int = 100) -> list[dict[str, Any]]:
        """返回最近 N 条副本；锁内复制后立即释放，避免 HTTP 序列化阻塞日志线程。"""
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
    """把标准 logging record 投递到内存缓冲，不改变其他 handler 的输出路径。"""
    def __init__(self, buffer: InMemoryLogBuffer) -> None:
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(record)


def setup_logging(settings, buffer_size: int = 200) -> InMemoryLogBuffer:
    """重置 root logger 并同时配置内存、控制台和 UTF-8 文件输出。"""
    level = getattr(logging, str(getattr(settings, "level", "INFO")).upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_buffer = InMemoryLogBuffer(capacity=buffer_size)

    # 内存 handler 始终启用；控制台由 YAML 控制，文件日志是运行证据的固定落点。
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
    # create_application 在测试或嵌入式调用中可能多次执行；先清理旧 handler 防止日志重复。
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    for handler in handlers:
        if handler.formatter is None:
            handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return log_buffer
