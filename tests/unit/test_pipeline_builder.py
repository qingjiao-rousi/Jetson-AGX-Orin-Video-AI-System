from __future__ import annotations

import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.infrastructure.pipeline.builder import PipelineBuilder
from app.settings import (
    AppSettings,
    DeepStreamSettings,
    LoggingSettings,
    OptimizationSettings,
    OutputSettings,
    SourceSettings,
)


class PipelineBuilderRuntimeTests(unittest.TestCase):
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
                    model_engine_path=Path("models/yolov8n.engine"),
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
            self.assertTrue(any(attachment["element"] == "tracker" for attachment in runtime["probe_attachments"]))
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
                    model_engine_path=Path("models/yolov8n.engine"),
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
                    "decodebin-2",
                    "nvvidconv-2",
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
            self.assertIn("pad-added", runtime["elements"]["decodebin-2"].handlers)
            self.assertIn("sink_0", runtime["elements"]["streammux"].requested)
            self.assertTrue(runtime["elements"]["tracker"].src_pad.probes)


if __name__ == "__main__":
    unittest.main()
