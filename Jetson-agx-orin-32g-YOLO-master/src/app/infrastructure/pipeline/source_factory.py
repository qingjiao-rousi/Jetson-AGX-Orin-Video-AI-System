from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Iterable

from app.settings import SourceSettings


@dataclass(frozen=True)
class SourceSpec:
    name: str
    uri: str
    kind: str
    enabled: bool

    @property
    def is_rtsp(self) -> bool:
        return self.kind == "rtsp"

    @property
    def is_file(self) -> bool:
        return self.kind == "file"


@dataclass(frozen=True)
class SourceNodeSpec:
    name: str
    element: str
    properties: dict[str, Any]
    stage: str
    flags: dict[str, Any]


@dataclass(frozen=True)
class SourceBranchSpec:
    source: SourceSpec
    nodes: tuple[SourceNodeSpec, ...]
    links: tuple[tuple[str, str], ...]
    mux_input: str


class SourceFactory:
    """
    Convert configured sources into a normalized source specification list.

    This layer does not create GStreamer objects yet. It only normalizes
    application inputs so the pipeline builder can consume a unified format.
    """

    def __init__(self, sources: Iterable[SourceSettings]) -> None:
        self._sources = tuple(sources)

    def list_sources(self) -> list[SourceSpec]:
        return [self._to_spec(source) for source in self._sources if source.enabled]

    def list_enabled_names(self) -> list[str]:
        return [spec.name for spec in self.list_sources()]

    def count_enabled(self) -> int:
        return len(self.list_sources())

    def build_branches(self) -> tuple[SourceBranchSpec, ...]:
        branches: list[SourceBranchSpec] = []
        for index, source in enumerate(self.list_sources()):
            branches.append(self._build_branch(index, source))
        return tuple(branches)

    def _to_spec(self, source: SourceSettings) -> SourceSpec:
        kind = source.kind.lower().strip()
        if kind not in {"rtsp", "file"}:
            raise ValueError(f"Unsupported source kind: {source.kind}")
        return SourceSpec(
            name=source.name,
            uri=self._normalize_uri(source.uri),
            kind=kind,
            enabled=source.enabled,
        )

    def _normalize_uri(self, uri: str) -> str:
        raw = uri.strip()
        if raw.startswith("file://"):
            return raw
        if "://" in raw:
            return raw
        if Path(raw).exists():
            return Path(raw).resolve().as_uri()
        return raw

    def _build_branch(self, index: int, source: SourceSpec) -> SourceBranchSpec:
        if source.is_rtsp:
            return self._build_rtsp_branch(index, source)
        return self._build_file_branch(index, source)

    def _build_rtsp_branch(self, index: int, source: SourceSpec) -> SourceBranchSpec:
        source_name = f"source-{index + 1}"
        depay_name = f"depay-{index + 1}"
        parser_name = f"parser-{index + 1}"
        decoder_name = f"decoder-{index + 1}"
        queue_name = f"pre-mux-queue-{index + 1}"

        nodes = (
            SourceNodeSpec(
                name=source_name,
                element="rtspsrc",
                properties={
                    "location": source.uri,
                    "latency": 200,
                    "drop-on-latency": True,
                    "protocols": "tcp",
                },
                stage="source",
                flags={
                    "required": True,
                    "live": True,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                    "dynamic_pad": True,
                    "reconnect_enabled": True,
                },
            ),
            SourceNodeSpec(
                name=depay_name,
                element="rtph264depay",
                properties={},
                stage="decode",
                flags={
                    "required": True,
                    "live": True,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                },
            ),
            SourceNodeSpec(
                name=parser_name,
                element="h264parse",
                properties={},
                stage="decode",
                flags={
                    "required": True,
                    "live": True,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                },
            ),
            SourceNodeSpec(
                name=decoder_name,
                element="nvv4l2decoder",
                properties={
                    "enable-max-performance": 1,
                    "drop-frame-interval": 0,
                },
                stage="decode",
                flags={
                    "required": True,
                    "live": True,
                    "supports_probe": False,
                    "hardware_accelerated": True,
                },
            ),
            SourceNodeSpec(
                name=queue_name,
                element="queue",
                properties={"max-size-buffers": 32},
                stage="buffer",
                flags={
                    "required": True,
                    "live": True,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                },
            ),
        )
        links = (
            (source_name, depay_name),
            (depay_name, parser_name),
            (parser_name, decoder_name),
            (decoder_name, queue_name),
        )
        return SourceBranchSpec(source=source, nodes=nodes, links=links, mux_input=queue_name)

    def _build_file_branch(self, index: int, source: SourceSpec) -> SourceBranchSpec:
        source_name = f"source-{index + 1}"
        demux_name = f"demux-{index + 1}"
        demux_queue_name = f"demux-queue-{index + 1}"
        parser_name = f"parser-{index + 1}"
        decoder_name = f"decoder-{index + 1}"
        convert_name = f"nvvidconv-{index + 1}"
        caps_name = f"source-caps-{index + 1}"
        queue_name = f"pre-mux-queue-{index + 1}"

        nodes = (
            SourceNodeSpec(
                name=source_name,
                element="filesrc",
                properties={"location": self._file_location(source.uri)},
                stage="source",
                flags={
                    "required": True,
                    "live": False,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                    "dynamic_pad": False,
                    "reconnect_enabled": False,
                },
            ),
            SourceNodeSpec(
                name=demux_name,
                element="qtdemux",
                properties={},
                stage="decode",
                flags={
                    "required": True,
                    "live": False,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                    "dynamic_pad": True,
                },
            ),
            SourceNodeSpec(
                name=demux_queue_name,
                element="queue",
                properties={},
                stage="decode",
                flags={
                    "required": True,
                    "live": False,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                },
            ),
            SourceNodeSpec(
                name=parser_name,
                element="h264parse",
                properties={},
                stage="decode",
                flags={
                    "required": True,
                    "live": False,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                },
            ),
            SourceNodeSpec(
                name=decoder_name,
                element="nvv4l2decoder",
                properties={},
                stage="decode",
                flags={
                    "required": True,
                    "live": False,
                    "supports_probe": False,
                    "hardware_accelerated": True,
                },
            ),
            SourceNodeSpec(
                name=convert_name,
                element="nvvideoconvert",
                properties={},
                stage="decode",
                flags={
                    "required": True,
                    "live": False,
                    "supports_probe": False,
                    "hardware_accelerated": True,
                },
            ),
            SourceNodeSpec(
                name=caps_name,
                element="capsfilter",
                properties={"caps": "video/x-raw(memory:NVMM),format=NV12"},
                stage="decode",
                flags={
                    "required": True,
                    "live": False,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                },
            ),
            SourceNodeSpec(
                name=queue_name,
                element="queue",
                properties={"max-size-buffers": 32},
                stage="buffer",
                flags={
                    "required": True,
                    "live": False,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                },
            ),
        )
        links = (
            (source_name, demux_name),
            (demux_name, demux_queue_name),
            (demux_queue_name, parser_name),
            (parser_name, decoder_name),
            (decoder_name, convert_name),
            (convert_name, caps_name),
            (caps_name, queue_name),
        )
        return SourceBranchSpec(source=source, nodes=nodes, links=links, mux_input=queue_name)

    def _file_location(self, uri: str) -> str:
        if uri.startswith("file://"):
            parsed = urlparse(uri)
            path = unquote(parsed.path)
            if parsed.netloc:
                path = f"//{parsed.netloc}{path}"
            return str(Path(path))
        return uri
