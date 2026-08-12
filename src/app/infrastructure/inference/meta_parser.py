from __future__ import annotations

"""将 DeepStream/C++/测试 mock 的松散 metadata 归一化为领域 ``FrameResult``。"""

from datetime import datetime, timedelta, timezone
from typing import Any

# 本层输出的 dataclass 是路由、指标和 JSONL 的共同输入，不携带 pyds 对象。
from app.domain.entities import BoundingBox, Detection, FrameResult, Track, canonical_stream_id


class MetaParser:
    """metadata 适配边界。

    上游可提供原生遍历转换结果、C++ JSON 归一化结果或测试字典；下游只面对
    UTC 时间戳、``stream-N``、Detection 与 Track。该类不访问 pyds，也不负责
    track ID 的跨路径重编号。
    """

    def parse(self, raw_meta: object) -> FrameResult:
        """兼容单帧调用；batch 输入时返回第一帧，批处理主路径应使用 ``parse_many``。"""
        results = self.parse_many(raw_meta)
        return results[0]

    def parse_many(self, raw_meta: object) -> list[FrameResult]:
        """将一个 DeepStream batch 展开为逐帧领域结果，避免把多路帧混成同一业务事件。"""
        # 先把 dict、native JSON 适配结果和测试 mock 归一化，再识别 batch/单帧结构。
        payload = self._normalize(raw_meta)
        frame_payloads = self._batch_frame_payloads(payload) or [payload]
        return [self._parse_frame(frame_payload, payload) for frame_payload in frame_payloads]

    def _parse_frame(self, frame_payload: dict[str, Any], payload: dict[str, Any]) -> FrameResult:
        """构造不可变业务快照；extra 只继承 batch 级扩展信息。"""
        # detections 保留所有主模型框；tracks 仅保留 tracker 有效 ID 的对象。
        return FrameResult(
            stream_id=self._parse_stream_id(frame_payload),
            frame_id=self._parse_frame_id(frame_payload),
            timestamp=self._parse_frame_timestamp(frame_payload),
            detections=self._parse_detections(frame_payload),
            tracks=self._parse_tracks(frame_payload),
            extra=dict(payload.get("extra", {})),
        )

    def _normalize(self, raw_meta: object) -> dict[str, Any]:
        if raw_meta is None:
            return {}
        if isinstance(raw_meta, dict):
            return raw_meta
        if hasattr(raw_meta, "__dict__"):
            return dict(vars(raw_meta))
        return {"raw": raw_meta}

    def _parse_frame_timestamp(self, payload: dict[str, Any]) -> datetime:
        """优先使用真实 wall-clock/NTP；PTS 仅在没有更好时钟时按相对纳秒解释。"""
        # 时间优先级避免把 PTS 误当墙钟：显式 timestamp > NTP epoch > 相对 PTS > 当前 UTC。
        if "timestamp" in payload and payload.get("timestamp") not in (None, ""):
            return self._parse_timestamp(payload.get("timestamp"), numeric_mode="auto")
        if "ntp_timestamp" in payload and payload.get("ntp_timestamp") not in (None, "", 0):
            return self._parse_timestamp(payload.get("ntp_timestamp"), numeric_mode="epoch")
        if "buf_pts" in payload and payload.get("buf_pts") not in (None, ""):
            return self._parse_timestamp(payload.get("buf_pts"), numeric_mode="relative_ns")
        return datetime.now(tz=timezone.utc)

    def _parse_timestamp(self, value: object, *, numeric_mode: str = "auto") -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            parsed = self._parse_timestamp_text(value)
            if parsed is not None:
                return parsed
            return datetime.now(tz=timezone.utc)
        if isinstance(value, (int, float)):
            return self._parse_numeric_timestamp(value, numeric_mode=numeric_mode)
        return datetime.now(tz=timezone.utc)

    def _parse_timestamp_text(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _parse_numeric_timestamp(self, value: int | float, *, numeric_mode: str) -> datetime:
        numeric = float(value)
        if numeric < 0:
            return datetime.now(tz=timezone.utc)

        if numeric_mode == "relative_ns":
            return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=numeric / 1_000_000_000)

        if numeric_mode == "epoch":
            return self._datetime_from_epoch_number(numeric)

        return self._datetime_from_epoch_number(numeric)

    def _datetime_from_epoch_number(self, value: float) -> datetime:
        """按量级兼容秒、毫秒、微秒、纳秒 epoch，非法值退化为当前 UTC 时间。"""
        if value >= 1e17:
            seconds = value / 1_000_000_000
        elif value >= 1e14:
            seconds = value / 1_000_000
        elif value >= 1e11:
            seconds = value / 1_000
        else:
            seconds = value
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return datetime.now(tz=timezone.utc)

    def _parse_stream_id(self, payload: dict[str, Any]) -> str:
        """统一 source_id、pad_index 与已有 stream-N 字符串，防止下游按名称分裂统计。"""
        if "stream_id" in payload:
            return canonical_stream_id(payload["stream_id"])
        if "source_id" in payload:
            return canonical_stream_id(payload["source_id"])
        if "pad_index" in payload:
            return canonical_stream_id(payload["pad_index"])
        return canonical_stream_id(None)

    def _parse_frame_id(self, payload: dict[str, Any]) -> int:
        if "frame_id" in payload:
            return int(payload["frame_id"])
        if "frame_num" in payload:
            return int(payload["frame_num"])
        return 0

    def _parse_detections(self, payload: dict[str, Any] | object) -> list[Detection]:
        """从 obj_meta_list/detections 读取所有检测；无 track 的新目标也应保留。"""
        detections: list[Detection] = []
        items = self._resolve_detection_items(payload)
        # 上游可给 bbox 或 DeepStream rect_params；都在此处统一为 left/top/width/height。
        for item in self._iterate_items(items):
            item_dict = self._item_to_dict(item)
            if item_dict is None:
                continue
            bbox = self._normalize_bbox(item_dict.get("bbox") or item_dict.get("rect_params", {}))
            detections.append(
                Detection(
                    class_id=int(item_dict.get("class_id", 0)),
                    class_name=str(item_dict.get("class_name") or item_dict.get("obj_label") or "unknown"),
                    confidence=float(item_dict.get("confidence", 0.0)),
                    bbox=BoundingBox(
                        left=float(bbox.get("left", 0.0)),
                        top=float(bbox.get("top", 0.0)),
                        width=float(bbox.get("width", 0.0)),
                        height=float(bbox.get("height", 0.0)),
                    ),
                )
            )
        return detections

    def _parse_tracks(self, payload: dict[str, Any] | object) -> list[Track]:
        """优先使用上游已生成的 tracks，否则只从具有效 ID 的 object metadata 构建。"""
        tracks: list[Track] = []
        items = self._resolve_track_items(payload)
        # Track 的 global_track_id 用于溯源，track_id 是 Builder 对本流映射后的业务 ID。
        for item in self._iterate_items(items):
            item_dict = self._item_to_dict(item)
            if item_dict is None:
                continue
            bbox = self._normalize_bbox(item_dict.get("bbox") or item_dict.get("rect_params", {}))
            tracks.append(
                Track(
                    track_id=int(item_dict.get("track_id", item_dict.get("object_id", 0))),
                    class_id=int(item_dict.get("class_id", 0)),
                    confidence=float(item_dict.get("confidence", 0.0)),
                    bbox=BoundingBox(
                        left=float(bbox.get("left", 0.0)),
                        top=float(bbox.get("top", 0.0)),
                        width=float(bbox.get("width", 0.0)),
                        height=float(bbox.get("height", 0.0)),
                    ),
                    global_track_id=self._parse_optional_int(
                        item_dict.get("global_track_id", item_dict.get("object_id"))
                    ),
                    class_name=str(item_dict.get("class_name") or item_dict.get("obj_label") or "unknown"),
                )
            )
        return tracks

    def _resolve_detection_items(self, payload: dict[str, Any] | object) -> object:
        if isinstance(payload, dict):
            return payload.get("detections", payload.get("obj_meta_list", []))
        return payload

    def _resolve_track_items(self, payload: dict[str, Any] | object) -> object:
        """未跟踪 object_id 为 DeepStream 哨兵值时不能进入 Track，避免伪稳定轨迹。"""
        if isinstance(payload, dict):
            tracks = payload.get("tracks")
            if tracks is not None:
                return tracks
            return [
                item
                for item in self._iterate_items(payload.get("obj_meta_list", []))
                if self._item_has_valid_track_id(item)
            ]
        return payload

    def _item_has_valid_track_id(self, item: object) -> bool:
        item_dict = self._item_to_dict(item)
        if item_dict is None:
            return False
        try:
            track_id = int(item_dict.get("track_id", item_dict.get("object_id", -1)))
        except (TypeError, ValueError):
            return False
        return track_id >= 0 and track_id != 0xFFFFFFFFFFFFFFFF

    def _parse_optional_int(self, value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _batch_frame_payloads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """兼容单帧 frame_meta 与 DeepStream GList/JSON batch 的 frame_meta_list。"""
        frame_meta = payload.get("frame_meta")
        if isinstance(frame_meta, dict):
            return [frame_meta]

        frame_meta_list = payload.get("frame_meta_list")
        if frame_meta_list is None:
            return []
        return [self._normalize(frame) for frame in self._iterate_items(frame_meta_list)]

    def _iterate_items(self, value: object):
        """统一 Python list、DeepStream GList 和单对象，隔离 pyds 链表细节。"""
        if isinstance(value, list):
            for item in value:
                yield item
            return

        if hasattr(value, "data") and hasattr(value, "next"):
            current = value
            while current is not None:
                yield getattr(current, "data", None)
                current = getattr(current, "next", None)
            return

        if value is not None:
            yield value

    def _item_to_dict(self, item: object) -> dict[str, Any] | None:
        if item is None:
            return None
        if isinstance(item, dict):
            return item
        if hasattr(item, "__dict__"):
            return dict(vars(item))
        return None

    def _normalize_bbox(self, value: object) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "__dict__"):
            data = dict(vars(value))
            return {
                "left": data.get("left", 0.0),
                "top": data.get("top", 0.0),
                "width": data.get("width", 0.0),
                "height": data.get("height", 0.0),
            }
        return {}
