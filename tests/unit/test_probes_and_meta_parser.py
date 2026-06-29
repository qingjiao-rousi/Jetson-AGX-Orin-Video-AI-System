from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.domain.entities import FrameResult
from app.infrastructure.inference.meta_parser import MetaParser
from app.infrastructure.pipeline.builder import PipelineBuilder
from app.infrastructure.pipeline.probes import ProbeRegistry
from app.settings import (
    AppSettings,
    DeepStreamSettings,
    LoggingSettings,
    OptimizationSettings,
    OutputSettings,
    SourceSettings,
)


class ProbeRegistryTests(unittest.TestCase):
    def test_emit_probe_payload_converts_raw_payload_into_frame_result(self) -> None:
        registry = ProbeRegistry()
        parser = MetaParser()
        captured = {}

        def handler(result) -> None:
            captured["result"] = result

        registry.register_frame_result_handler(handler)

        payload = {
            "stream_id": "stream-2",
            "frame_id": 10,
            "timestamp": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "detections": [
                {
                    "class_id": 1,
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox": {"left": 10, "top": 20, "width": 30, "height": 40},
                }
            ],
        }

        registry.emit_probe_payload(payload, parser)

        self.assertIn("result", captured)
        self.assertEqual(captured["result"].stream_id, "stream-2")
        self.assertEqual(len(captured["result"].detections), 1)


class MetaParserTests(unittest.TestCase):
    def test_parse_supports_deepstream_style_object_meta(self) -> None:
        parser = MetaParser()
        payload = {
            "source_id": 3,
            "frame_num": 15,
            "ntp_timestamp": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "obj_meta_list": [
                {
                    "class_id": 2,
                    "obj_label": "car",
                    "confidence": 0.85,
                    "rect_params": {
                        "left": 100,
                        "top": 50,
                        "width": 60,
                        "height": 40,
                    },
                    "object_id": 99,
                }
            ],
        }

        result = parser.parse(payload)

        self.assertEqual(result.stream_id, "stream-3")
        self.assertEqual(result.frame_id, 15)
        self.assertEqual(result.detections[0].class_name, "car")
        self.assertEqual(result.tracks[0].track_id, 99)

    def test_parse_supports_batch_meta_like_object_graph(self) -> None:
        parser = MetaParser()

        class FakeNode:
            def __init__(self, data, next_node=None) -> None:
                self.data = data
                self.next = next_node

        class FakeRect:
            def __init__(self, left, top, width, height) -> None:
                self.left = left
                self.top = top
                self.width = width
                self.height = height

        class FakeObjectMeta:
            def __init__(self, class_id, label, confidence, object_id, rect) -> None:
                self.class_id = class_id
                self.obj_label = label
                self.confidence = confidence
                self.object_id = object_id
                self.rect_params = rect

        class FakeFrameMeta:
            def __init__(self) -> None:
                self.pad_index = 4
                self.frame_num = 22
                self.ntp_timestamp = datetime(2026, 6, 23, tzinfo=timezone.utc)
                obj = FakeObjectMeta(5, "bus", 0.76, 123, FakeRect(1, 2, 3, 4))
                self.obj_meta_list = FakeNode(obj)

        class FakeBatchMeta:
            def __init__(self) -> None:
                self.frame_meta_list = FakeNode(FakeFrameMeta())

        result = parser.parse(FakeBatchMeta())

        self.assertEqual(result.stream_id, "stream-4")
        self.assertEqual(result.frame_id, 22)
        self.assertEqual(result.detections[0].class_name, "bus")
        self.assertEqual(result.tracks[0].track_id, 123)


class BuilderProbeDispatchTests(unittest.TestCase):
    def test_probe_buffer_dispatches_payload_to_registry(self) -> None:
        settings = AppSettings(
            app_name="deepstream-multistream",
            source_count=1,
            sources=(
                SourceSettings(name="cam1", uri="rtsp://127.0.0.1:8554/stream1", kind="rtsp", enabled=True),
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
        parser = MetaParser()
        registry = ProbeRegistry()
        captured: dict[str, FrameResult] = {}

        def handler(result: FrameResult) -> None:
            captured["result"] = result

        registry.register_frame_result_handler(handler)
        payload = {
            "stream_id": "stream-1",
            "frame_id": 7,
            "detections": [],
            "tracks": [],
        }

        builder._on_probe_buffer(None, payload, {"probe_registry": registry, "meta_parser": parser})

        self.assertIn("result", captured)
        self.assertEqual(captured["result"].frame_id, 7)

    def test_builder_probe_prefers_buffer_batch_meta_when_available(self) -> None:
        settings = AppSettings(
            app_name="deepstream-multistream",
            source_count=1,
            sources=(
                SourceSettings(name="cam1", uri="rtsp://127.0.0.1:8554/stream1", kind="rtsp", enabled=True),
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
        parser = MetaParser()
        registry = ProbeRegistry()
        captured: dict[str, FrameResult] = {}

        def handler(result: FrameResult) -> None:
            captured["result"] = result

        registry.register_frame_result_handler(handler)

        class FakeBuffer:
            pass

        class FakeInfo:
            def get_buffer(self):
                return FakeBuffer()

        class FakePyds:
            def gst_buffer_get_nvds_batch_meta(self, _):
                class FakeNode:
                    def __init__(self, data, next_node=None) -> None:
                        self.data = data
                        self.next = next_node

                class FakeRect:
                    def __init__(self) -> None:
                        self.left = 7
                        self.top = 8
                        self.width = 9
                        self.height = 10

                class FakeObject:
                    def __init__(self) -> None:
                        self.class_id = 9
                        self.obj_label = "cat"
                        self.confidence = 0.66
                        self.object_id = 77
                        self.rect_params = FakeRect()

                class FakeFrame:
                    def __init__(self) -> None:
                        self.pad_index = 2
                        self.frame_num = 44
                        self.ntp_timestamp = datetime(2026, 6, 23, tzinfo=timezone.utc)
                        self.obj_meta_list = FakeNode(FakeObject())

                class FakeBatch:
                    def __init__(self) -> None:
                        self.frame_meta_list = FakeNode(FakeFrame())

                return FakeBatch()

        builder._runtime_factory._pyds = FakePyds()
        builder._on_probe_buffer(None, FakeInfo(), {"probe_registry": registry, "meta_parser": parser})

        self.assertEqual(captured["result"].stream_id, "stream-2")
        self.assertEqual(captured["result"].detections[0].class_name, "cat")


if __name__ == "__main__":
    unittest.main()
