from __future__ import annotations

import unittest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.infrastructure.pipeline.builder import GStreamerRuntimeFactory, PipelineBuilder
from app.settings import (
    AppSettings,
    DeepStreamSettings,
    LoggingSettings,
    OptimizationSettings,
    OutputSettings,
    SourceSettings,
)


class PipelineBuilderRuntimeTests(unittest.TestCase):
    def test_gstreamer_import_failure_is_logged(self) -> None:
        factory = GStreamerRuntimeFactory.__new__(GStreamerRuntimeFactory)
        factory._gst = object()
        factory._available = True

        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "gi":
                raise RuntimeError("gi unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertLogs(level="WARNING") as logs:
                factory._load()

        self.assertIsNone(factory._gst)
        self.assertFalse(factory._available)
        self.assertTrue(any("GStreamer runtime unavailable" in entry for entry in logs.output))

    def test_pyds_import_failure_is_logged(self) -> None:
        factory = GStreamerRuntimeFactory.__new__(GStreamerRuntimeFactory)
        factory._pyds = object()

        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "pyds":
                raise RuntimeError("pyds unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertLogs(level="WARNING") as logs:
                factory._load_pyds()

        self.assertIsNone(factory._pyds)
        self.assertTrue(any("DeepStream pyds bindings unavailable" in entry for entry in logs.output))

    def test_build_runtime_exposes_static_and_dynamic_link_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample.mp4"
            sample.write_bytes(b"")
            settings = AppSettings(
                app_name="deepstream-multistream",
                source_count=2,
                sources=(
                    SourceSettings(name="cam1", uri="rtsp://127.0.0.1:8554/stream1", kind="rtsp", enabled=True),
                    SourceSettings(name="cam2", uri=str(sample), kind="file", enabled=True),
                ),
                logging=LoggingSettings(),
                output=OutputSettings(enable_jsonl=True),
                optimization=OptimizationSettings(),
                deepstream=DeepStreamSettings(
                    model_engine_path=Path("models/yolov8s.engine"),
                    custom_lib_path=Path("custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"),
                    tracker_config_path=Path("configs/deepstream/tracker_iou.yml"),
                    infer_config_path=Path("configs/deepstream/infer_primary_yolo.txt"),
                    streammux_config_path=Path("configs/deepstream/streammux.yaml"),
                ),
            )

            builder = PipelineBuilder(settings)

            runtime = builder.build_runtime()

            self.assertIn("blueprint", runtime)
            self.assertIn("static_links", runtime)
            self.assertIn("dynamic_links", runtime)
            self.assertIn("streammux_requests", runtime)
            self.assertIn("probe_attachments", runtime)
            self.assertIn("assembly_steps", runtime)
            self.assertIn(("streammux", "primary-infer"), runtime["static_links"])
            self.assertTrue(any(plan["source"] == "source-1" for plan in runtime["dynamic_links"]))
            self.assertTrue(any(req["target"] == "streammux" for req in runtime["streammux_requests"]))
            self.assertTrue(any(attachment["element"] == "osd" for attachment in runtime["probe_attachments"]))
            self.assertIn("add_elements_to_pipeline", runtime["assembly_steps"])

    def test_runtime_helpers_bind_dynamic_pads_and_probes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample.mp4"
            sample.write_bytes(b"")
            settings = AppSettings(
                app_name="deepstream-multistream",
                source_count=2,
                sources=(
                    SourceSettings(name="cam1", uri="rtsp://127.0.0.1:8554/stream1", kind="rtsp", enabled=True),
                    SourceSettings(name="cam2", uri=str(sample), kind="file", enabled=True),
                ),
                logging=LoggingSettings(),
                output=OutputSettings(enable_jsonl=True),
                optimization=OptimizationSettings(),
                deepstream=DeepStreamSettings(
                    model_engine_path=Path("models/yolov8s.engine"),
                    custom_lib_path=Path("custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"),
                    tracker_config_path=Path("configs/deepstream/tracker_iou.yml"),
                    infer_config_path=Path("configs/deepstream/infer_primary_yolo.txt"),
                    streammux_config_path=Path("configs/deepstream/streammux.yaml"),
                ),
            )
            builder = PipelineBuilder(settings)
            runtime = builder.build_runtime()

            class FakePad:
                def __init__(self) -> None:
                    self.linked_to = None
                    self.probes = []

                def link(self, other) -> int:
                    self.linked_to = other
                    return 0

                def add_probe(self, probe_type, callback) -> int:
                    self.probes.append((probe_type, callback))
                    return len(self.probes)

            class FakeElement:
                def __init__(self, name: str) -> None:
                    self.name = name
                    self.links = []
                    self.handlers = {}
                    self.src_pad = FakePad()
                    self.sink_pad = FakePad()
                    self.requested = {}

                def link(self, other) -> bool:
                    self.links.append(other.name)
                    return True

                def connect(self, signal_name, callback, user_data) -> int:
                    self.handlers[signal_name] = (callback, user_data)
                    return 1

                def get_static_pad(self, pad_name: str):
                    return self.src_pad if pad_name == "src" else self.sink_pad

                def get_request_pad(self, pad_name: str):
                    pad = FakePad()
                    self.requested[pad_name] = pad
                    return pad

            runtime["elements"] = {
                name: FakeElement(name)
                for name in (
                    "source-1",
                    "depay-1",
                    "parser-1",
                    "decoder-1",
                    "pre-mux-queue-1",
                    "source-2",
                    "demux-2",
                    "demux-queue-2",
                    "parser-2",
                    "decoder-2",
                    "nvvidconv-2",
                    "source-caps-2",
                    "pre-mux-queue-2",
                    "streammux",
                    "primary-infer",
                    "tracker",
                    "osd",
                    "post-osd-queue",
                    "encoder",
                    "rtmp-pay",
                    "sink",
                )
            }

            builder.prepare_dynamic_pad_handlers(runtime)
            builder.prepare_streammux_requests(runtime)
            builder.attach_probe_points(runtime)

            self.assertIn("pad-added", runtime["elements"]["source-1"].handlers)
            self.assertIn("pad-added", runtime["elements"]["demux-2"].handlers)
            self.assertIn("sink_0", runtime["elements"]["streammux"].requested)
            self.assertTrue(runtime["elements"]["osd"].sink_pad.probes)

    def test_multisource_tiler_is_inserted_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample_1 = Path(tmp) / "sample-1.mp4"
            sample_2 = Path(tmp) / "sample-2.mp4"
            sample_1.write_bytes(b"")
            sample_2.write_bytes(b"")
            settings = AppSettings(
                app_name="deepstream-multifile",
                source_count=2,
                sources=(
                    SourceSettings(name="local1", uri=str(sample_1), kind="file", enabled=True),
                    SourceSettings(name="local2", uri=str(sample_2), kind="file", enabled=True),
                ),
                logging=LoggingSettings(),
                output=OutputSettings(enable_jsonl=True),
                optimization=OptimizationSettings(),
                deepstream=DeepStreamSettings(
                    batch_size=2,
                    enable_tiler=True,
                    tiler_rows=1,
                    tiler_columns=2,
                    tiler_width=1280,
                    tiler_height=720,
                    output_sink="file",
                    output_video_path=Path(tmp) / "tiled.mp4",
                    model_engine_path=Path("models/yolov8s.engine"),
                    custom_lib_path=Path("custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"),
                    tracker_config_path=Path("configs/deepstream/tracker_iou.yml"),
                    infer_config_path=Path("configs/deepstream/infer_primary_yolo.txt"),
                    streammux_config_path=Path("configs/deepstream/streammux.yaml"),
                ),
            )

            blueprint = PipelineBuilder(settings).build()
            node_by_name = {node.name: node for node in blueprint.nodes}

            self.assertIn("tiler", node_by_name)
            self.assertEqual(node_by_name["tiler"].element, "nvmultistreamtiler")
            self.assertEqual(node_by_name["tiler"].properties["rows"], 1)
            self.assertEqual(node_by_name["tiler"].properties["columns"], 2)
            self.assertIn(("tracker", "tiler"), blueprint.links)
            self.assertIn(("tiler", "pre-osd-convert"), blueprint.links)
            self.assertEqual(blueprint.probes, (("tiler", "sink"),))

    def test_rtsp_output_uses_h264_payloader_and_configured_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample.mp4"
            sample.write_bytes(b"sample")
            settings = AppSettings(
                app_name="deepstream-rtsp-output",
                source_count=1,
                sources=(SourceSettings(name="local", uri=str(sample), kind="file", enabled=True),),
                logging=LoggingSettings(),
                output=OutputSettings(enable_jsonl=True),
                optimization=OptimizationSettings(),
                deepstream=DeepStreamSettings(
                    batch_size=1,
                    output_sink="rtsp",
                    output_url="rtsp://127.0.0.1:8554/inference",
                    model_engine_path=Path("models/yolov8s.engine"),
                    custom_lib_path=Path("custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"),
                    tracker_config_path=Path("configs/deepstream/tracker_iou.yml"),
                    infer_config_path=Path("configs/deepstream/infer_primary_yolo.txt"),
                    streammux_config_path=Path("configs/deepstream/streammux.yaml"),
                ),
            )

            blueprint = PipelineBuilder(settings).build()
            nodes = {node.name: node for node in blueprint.nodes}

            self.assertEqual(nodes["sink"].element, "rtspclientsink")
            self.assertEqual(nodes["sink"].properties["location"], "rtsp://127.0.0.1:8554/inference")
            self.assertEqual(nodes["rtsp-pay"].element, "rtph264pay")
            self.assertIn(("h264-parser", "rtsp-pay"), blueprint.links)
            self.assertIn(("rtsp-pay", "sink"), blueprint.links)

    def test_rtmp_output_uses_configured_url(self) -> None:
        settings = AppSettings(
            app_name="deepstream-rtmp-output",
            source_count=1,
            sources=(SourceSettings(name="cam", uri="rtsp://127.0.0.1:8554/stream1", kind="rtsp", enabled=True),),
            logging=LoggingSettings(),
            output=OutputSettings(enable_jsonl=True),
            optimization=OptimizationSettings(),
            deepstream=DeepStreamSettings(
                batch_size=1,
                output_sink="rtmp",
                output_url="rtmp://127.0.0.1:1935/live/inference",
            ),
        )

        blueprint = PipelineBuilder(settings).build()
        nodes = {node.name: node for node in blueprint.nodes}

        self.assertEqual(nodes["sink"].element, "rtmpsink")
        self.assertEqual(nodes["sink"].properties["location"], "rtmp://127.0.0.1:1935/live/inference")
        self.assertEqual(nodes["output-mux"].element, "flvmux")


if __name__ == "__main__":
    unittest.main()
