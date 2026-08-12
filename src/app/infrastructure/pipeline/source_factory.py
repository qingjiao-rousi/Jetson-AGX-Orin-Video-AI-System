from __future__ import annotations

"""将应用层输入流声明转换为可拼接到 DeepStream 主图的 source branch。"""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Iterable

# SourceSettings 是应用配置；以下 Spec 只描述 GStreamer 子图，不创建运行时元素。
from app.settings import SourceSettings


@dataclass(frozen=True)
class SourceSpec:
    """已归一化的输入流；URI 不再区分相对路径与 file URI。"""
    name: str
    uri: str
    kind: str
    enabled: bool
    scene: str = "normal"
    priority: str = "medium"
    zones: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    @property
    def is_rtsp(self) -> bool:
        return self.kind == "rtsp"

    @property
    def is_file(self) -> bool:
        return self.kind == "file"


@dataclass(frozen=True)
class SourceNodeSpec:
    """单个 source 分支元素的声明，不直接持有 GStreamer 对象。"""
    name: str
    element: str
    properties: dict[str, Any]
    stage: str
    flags: dict[str, Any]


@dataclass(frozen=True)
class SourceBranchSpec:
    """从一个输入到 ``nvstreammux`` 前队列的完整子图。"""
    source: SourceSpec
    nodes: tuple[SourceNodeSpec, ...]
    links: tuple[tuple[str, str], ...]
    mux_input: str


class SourceFactory:
    """把配置源转为统一的 branch 声明，不在此层创建 GStreamer 元素。

    RTSP 与 MP4 的解复用方式不同，但二者最终均输出 NVMM/NV12，并通过独立
    ``pre-mux-queue`` 接入 ``nvstreammux``，因此主检测链路无需关心输入类型。
    """

    def __init__(self, sources: Iterable[SourceSettings]) -> None:
        self._sources = tuple(sources)

    def list_sources(self) -> list[SourceSpec]:
        """过滤禁用 source 并归一化 URI/输入类型，顺序即后续 streammux pad 编号。"""
        return [self._to_spec(source) for source in self._sources if source.enabled]

    def list_enabled_names(self) -> list[str]:
        return [spec.name for spec in self.list_sources()]

    def count_enabled(self) -> int:
        return len(self.list_sources())

    def build_branches(self) -> tuple[SourceBranchSpec, ...]:
        """为每个启用 source 生成独立解码支路；序号同时决定 streammux sink_N。"""
        branches: list[SourceBranchSpec] = []
        for index, source in enumerate(self.list_sources()):
            branches.append(self._build_branch(index, source))
        return tuple(branches)

    def _to_spec(self, source: SourceSettings) -> SourceSpec:
        """校验输入种类，并保留场景/能力等供 Builder describe 与路由对齐的元数据。"""
        kind = source.kind.lower().strip()
        if kind not in {"rtsp", "file"}:
            raise ValueError(f"Unsupported source kind: {source.kind}")
        return SourceSpec(
            name=source.name,
            uri=self._normalize_uri(source.uri),
            kind=kind,
            enabled=source.enabled,
            scene=source.scene,
            priority=source.priority,
            zones=source.zones,
            capabilities=source.capabilities,
        )

    def _normalize_uri(self, uri: str) -> str:
        """把存在的本地相对路径转为 file URI，保留网络 URI 与尚待运行时解析的路径。"""
        raw = uri.strip()
        if raw.startswith("file://"):
            return raw
        if "://" in raw:
            return raw
        if Path(raw).exists():
            return Path(raw).resolve().as_uri()
        return raw

    def _build_branch(self, index: int, source: SourceSpec) -> SourceBranchSpec:
        """按 source 类型选择解码图；两条支路最终都产出 NVMM/NV12 给 streammux。"""
        if source.is_rtsp:
            return self._build_rtsp_branch(index, source)
        return self._build_file_branch(index, source)

    def _build_rtsp_branch(self, index: int, source: SourceSpec) -> SourceBranchSpec:
        """构建 live RTSP 解码支路。

        采用 TCP、有限 latency 和 ``drop-on-latency``：实时系统优先新帧，网络抖动
        时允许丢失过期帧而不无限累计延迟。
        """
        source_name = f"source-{index + 1}"
        depay_name = f"depay-{index + 1}"
        parser_name = f"parser-{index + 1}"
        decoder_name = f"decoder-{index + 1}"
        convert_name = f"nvvidconv-{index + 1}"
        caps_name = f"source-caps-{index + 1}"
        queue_name = f"pre-mux-queue-{index + 1}"

        # rtspsrc 与 qtdemux 都在运行时产生 src pad，因此第一个链接由 pad-added 完成。
        nodes = (
            SourceNodeSpec(
                name=source_name,
                element="rtspsrc",
                properties={
                    "location": source.uri,
                    "latency": 200,
                    # 实时流优先新帧，网络抖动时宁可丢旧帧也不无限增长排队延迟。
                    "drop-on-latency": True,
                    "protocols": "tcp",
                    "do-rtsp-keep-alive": True,
                    "retry": 5,
                    "timeout": 5_000_000,
                    "tcp-timeout": 5_000_000,
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
                name=convert_name,
                element="nvvideoconvert",
                properties={},
                stage="decode",
                flags={
                    "required": True,
                    "live": True,
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
                    "live": True,
                    "supports_probe": False,
                    "hardware_accelerated": False,
                },
            ),
            SourceNodeSpec(
                name=queue_name,
                element="queue",
                properties={
                    "max-size-buffers": 32,
                    "max-size-time": 0,
                    "max-size-bytes": 0,
                },
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
            (decoder_name, convert_name),
            (convert_name, caps_name),
            (caps_name, queue_name),
        )
        return SourceBranchSpec(source=source, nodes=nodes, links=links, mux_input=queue_name)

    def _build_file_branch(self, index: int, source: SourceSpec) -> SourceBranchSpec:
        """构建 MP4 文件支路；qtdemux 的动态 pad 在 runtime 阶段连接到解码队列。"""
        source_name = f"source-{index + 1}"
        demux_name = f"demux-{index + 1}"
        demux_queue_name = f"demux-queue-{index + 1}"
        parser_name = f"parser-{index + 1}"
        decoder_name = f"decoder-{index + 1}"
        convert_name = f"nvvidconv-{index + 1}"
        caps_name = f"source-caps-{index + 1}"
        queue_name = f"pre-mux-queue-{index + 1}"

        # 文件输入没有网络重连语义，但 qtdemux 同样需要动态 pad 连接到后续队列。
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
