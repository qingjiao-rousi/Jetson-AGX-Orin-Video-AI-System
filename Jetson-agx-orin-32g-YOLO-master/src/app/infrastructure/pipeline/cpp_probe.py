from __future__ import annotations

import ctypes
import json
import logging
import os
from pathlib import Path
from typing import Any


class CppProbeHandler:
    """Optional ctypes bridge for the native NvDsBatchMeta parser."""

    def __init__(self, library_path: Path | None = None) -> None:
        self.library_path = library_path or Path("build/probe_handler/libprobe_handler.so")
        self._library: Any | None = None
        self._parse = None
        self._parse_buffer = None
        self._free = None
        self._load()

    @property
    def available(self) -> bool:
        return self._library is not None and self._parse is not None and self._free is not None

    def _load(self) -> None:
        project_root = Path(__file__).resolve().parents[4]
        candidates: list[Path] = []
        env_path = os.environ.get("CPP_PROBE_HANDLER_PATH")
        if env_path:
            raw_env_path = Path(env_path)
            candidates.extend(
                [raw_env_path, project_root / raw_env_path]
                if not raw_env_path.is_absolute()
                else [raw_env_path]
            )
        raw_library_path = Path(self.library_path)
        candidates.extend(
            [raw_library_path, project_root / raw_library_path]
            if not raw_library_path.is_absolute()
            else [raw_library_path]
        )

        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if not candidate.exists():
                continue
            try:
                library = ctypes.CDLL(str(candidate))
                parse = library.probe_parse_nvds_batch_meta_json
                free = library.probe_free_json
                parse.argtypes = [ctypes.c_void_p]
                parse.restype = ctypes.c_void_p
                free.argtypes = [ctypes.c_void_p]
                free.restype = None
                parse_buffer = getattr(library, "probe_parse_gst_buffer_json", None)
                if parse_buffer is not None:
                    parse_buffer.argtypes = [ctypes.c_void_p]
                    parse_buffer.restype = ctypes.c_void_p
                self._library = library
                self._parse = parse
                self._parse_buffer = parse_buffer
                self._free = free
                logging.info("native C++ probe parser enabled: %s", candidate)
                return
            except (OSError, AttributeError) as exc:
                logging.warning("failed to load native C++ probe parser `%s`: %s", candidate, exc)

        logging.info("native C++ probe parser unavailable; using Python metadata traversal")

    def parse(self, batch_meta: object) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("native C++ probe parser is unavailable")

        try:
            pointer = hash(batch_meta)
        except Exception as exc:
            raise RuntimeError("failed to obtain NvDsBatchMeta pointer") from exc
        if pointer <= 0:
            raise RuntimeError("invalid NvDsBatchMeta pointer")

        raw_pointer = self._parse(ctypes.c_void_p(pointer))
        if not raw_pointer:
            raise RuntimeError("native C++ probe parser returned a null payload")

        try:
            raw_json = ctypes.string_at(raw_pointer).decode("utf-8")
        finally:
            self._free(raw_pointer)

        try:
            frames = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("native C++ probe parser returned invalid JSON") from exc
        return self.to_batch_payload(frames)

    def parse_buffer(self, buffer: object) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("native C++ probe parser is unavailable")
        if not self.is_gst_buffer(buffer):
            raise RuntimeError("probe object is not a real Gst.Buffer")
        parse_buffer = self._parse_buffer
        if parse_buffer is None:
            raise RuntimeError("native C++ GST buffer parser symbol is unavailable")

        try:
            pointer = hash(buffer)
        except Exception as exc:
            raise RuntimeError("failed to obtain GstBuffer pointer") from exc
        if pointer <= 0:
            raise RuntimeError("invalid GstBuffer pointer")
        return self._parse_pointer(parse_buffer, pointer)

    @staticmethod
    def is_gst_buffer(buffer: object) -> bool:
        module = type(buffer).__module__
        class_name = type(buffer).__name__
        if class_name == "Buffer" and "Gst" in module:
            return True
        gtype = getattr(buffer, "__gtype__", None)
        gtype_name = getattr(gtype, "name", "")
        return class_name == "Buffer" and "GstBuffer" in str(gtype_name)

    def _parse_pointer(self, parse_function: Any, pointer: int) -> dict[str, Any]:
        raw_pointer = parse_function(ctypes.c_void_p(pointer))
        if not raw_pointer:
            raise RuntimeError("native C++ probe parser returned a null payload")
        try:
            raw_json = ctypes.string_at(raw_pointer).decode("utf-8")
        finally:
            self._free(raw_pointer)
        try:
            frames = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("native C++ probe parser returned invalid JSON") from exc
        return self.to_batch_payload(frames)

    @staticmethod
    def to_batch_payload(frames: object) -> dict[str, Any]:
        """Normalize the native frame array to the existing MetaParser contract."""
        if not isinstance(frames, list):
            raise RuntimeError("native C++ probe payload must be a JSON array")

        normalized_frames: list[dict[str, Any]] = []
        for raw_frame in frames:
            if not isinstance(raw_frame, dict):
                continue
            source_id = raw_frame.get("source_id", raw_frame.get("stream_id", 0))
            objects: list[dict[str, Any]] = []
            for raw_detection in raw_frame.get("detections", []):
                if not isinstance(raw_detection, dict):
                    continue
                track_id = raw_detection.get("track_id", 0)
                objects.append(
                    {
                        "class_id": raw_detection.get("class_id", 0),
                        "obj_label": raw_detection.get("class_name", "unknown"),
                        "confidence": raw_detection.get("confidence", 0.0),
                        "track_id": track_id,
                        "object_id": track_id,
                        "global_track_id": track_id,
                        "rect_params": raw_detection.get("bbox", {}),
                    }
                )
            normalized_frames.append(
                {
                    "stream_id": source_id,
                    "source_id": source_id,
                    "frame_id": raw_frame.get("frame_id", 0),
                    "frame_num": raw_frame.get("frame_id", 0),
                    "ntp_timestamp": raw_frame.get("ntp_timestamp", 0),
                    "obj_meta_list": objects,
                    "tracks": objects,
                }
            )
        return {"frame_meta_list": normalized_frames}
