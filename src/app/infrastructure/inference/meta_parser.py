from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.domain.entities import BoundingBox, Detection, FrameResult, Track


class MetaParser:
    def parse(self, raw_meta: object) -> FrameResult:
        payload = self._normalize(raw_meta)
        batch_payload = self._unwrap_batch_payload(payload)
        frame_payload = batch_payload or payload
        detections = self._parse_detections(frame_payload)
        tracks = self._parse_tracks(frame_payload)
        return FrameResult(
            stream_id=self._parse_stream_id(frame_payload),
            frame_id=self._parse_frame_id(frame_payload),
            timestamp=self._parse_timestamp(
                frame_payload.get("timestamp")
                or frame_payload.get("ntp_timestamp")
                or frame_payload.get("buf_pts")
            ),
            detections=detections,
            tracks=tracks,
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

    def _parse_timestamp(self, value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.now(tz=timezone.utc)

    def _parse_stream_id(self, payload: dict[str, Any]) -> str:
        if "stream_id" in payload:
            value = payload["stream_id"]
            if isinstance(value, int):
                return f"stream-{value}"
            text = str(value)
            if text.isdigit():
                return f"stream-{text}"
            return text
        if "source_id" in payload:
            return f"stream-{payload['source_id']}"
        if "pad_index" in payload:
            return f"stream-{payload['pad_index']}"
        return "stream-0"

    def _parse_frame_id(self, payload: dict[str, Any]) -> int:
        if "frame_id" in payload:
            return int(payload["frame_id"])
        if "frame_num" in payload:
            return int(payload["frame_num"])
        return 0

    def _parse_detections(self, payload: dict[str, Any] | object) -> list[Detection]:
        detections: list[Detection] = []
        items = self._resolve_detection_items(payload)
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
        tracks: list[Track] = []
        items = self._resolve_track_items(payload)
        for item in self._iterate_items(items):
            item_dict = self._item_to_dict(item)
            if item_dict is None:
                continue
            bbox = self._normalize_bbox(item_dict.get("bbox") or item_dict.get("rect_params", {}))
            tracks.append(
                Track(
                    track_id=int(item_dict.get("track_id", item_dict.get("object_id", 0))),
                    class_id=int(item_dict.get("class_id", 0)),
                    bbox=BoundingBox(
                        left=float(bbox.get("left", 0.0)),
                        top=float(bbox.get("top", 0.0)),
                        width=float(bbox.get("width", 0.0)),
                        height=float(bbox.get("height", 0.0)),
                    ),
                )
            )
        return tracks

    def _resolve_detection_items(self, payload: dict[str, Any] | object) -> object:
        if isinstance(payload, dict):
            return payload.get("detections", payload.get("obj_meta_list", []))
        return payload

    def _resolve_track_items(self, payload: dict[str, Any] | object) -> object:
        if isinstance(payload, dict):
            return payload.get("tracks", payload.get("obj_meta_list", []))
        return payload

    def _unwrap_batch_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        frame_meta = payload.get("frame_meta")
        if isinstance(frame_meta, dict):
            return frame_meta

        frame_meta_list = payload.get("frame_meta_list")
        if frame_meta_list is None:
            return None

        first_frame = self._first_from_iterable(frame_meta_list)
        if first_frame is None:
            return None

        return self._normalize(first_frame)

    def _first_from_iterable(self, value: object) -> object | None:
        if isinstance(value, list):
            return value[0] if value else None
        if hasattr(value, "data") and hasattr(value, "next"):
            return getattr(value, "data", None)
        return None

    def _iterate_items(self, value: object):
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
