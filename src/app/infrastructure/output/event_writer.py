from __future__ import annotations

"""将异步专用模型和场景规则事件写入独立 JSONL。"""

# worker 事件可为 dataclass，场景规则事件为 dict；在此统一为 JSONL。
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from threading import Lock
from typing import Any


class EventWriter:
    """线程安全的事件 writer。

    Worker 与主线程都可能调用它，因此写入和 flush 在互斥锁内完成；与主结果 writer
    不同，它没有队列，事件量应由路由、确认帧和规则节流控制。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        self._lock = Lock()
        self._written = 0
        self._errors = 0

    def write(self, event: object) -> None:
        """序列化 dataclass 或字典事件并追加一行；错误交给调用方记录/处理。"""
        payload = asdict(event) if is_dataclass(event) else dict(event) if isinstance(event, dict) else event
        try:
            line = json.dumps(payload, ensure_ascii=False, default=self._json_default)
            # 多个 worker 可并发写事件；锁覆盖 write+flush，避免一行 JSON 被交叉写坏。
            with self._lock:
                self._file.write(line + "\n")
                self._file.flush()
                self._written += 1
        except Exception:
            with self._lock:
                self._errors += 1
            raise

    def stats(self) -> dict[str, Any]:
        """返回持久化成功与错误次数，不推断事件业务正确性。"""
        with self._lock:
            return {
                "path": str(self._path),
                "written": self._written,
                "errors": self._errors,
            }

    def close(self) -> None:
        """在所有 worker 停止后由 Orchestrator 调用，关闭事件文件。"""
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
