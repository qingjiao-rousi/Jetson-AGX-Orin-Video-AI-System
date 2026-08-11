# Model Assets

This repository tracks labels and model-side configuration only. It does not
include ONNX files, TensorRT engines, calibration images, weights, or videos.

Prepare the following local assets before running the full pipeline:

| Asset | Expected local path | Notes |
| --- | --- | --- |
| Primary YOLO ONNX | `export_yolov8_ds/yolov8s.onnx` | Source for the current primary INT8 engine build; its output is `[batch,8400,6]`. |
| Primary FP16 engine | `models/fp16/yolov8s.engine` | Build with `scripts/build_fp16_engines.sh --primary-only`. |
| Primary INT8 candidate engine | `models/int8/yolov8s_coco_train504.engine` | Current validated candidate; build with `scripts/build_primary_detector_int8.py` and the independent COCO train504 calibration set. |
| Specialist FP16 engines | `models/fp16/` | Build with `scripts/build_fp16_engines.sh --specialists-only`. |
| Candidate calibration images | `calibration/coco_train504/images/` | Independent COCO train2017 calibration set used by the current INT8 candidate. |
| Local videos | `video/1.mp4` through `video/8.mp4` | Required by the bundled multi-file examples. |
| COCO val2017 (optional) | `datasets/coco/` | Public, local-only labeled evaluation data; see `docs/coco_fp16_int8_evaluation.md`. |

TensorRT engines are specific to the target Jetson software and hardware
environment. Build them on the deployment device after confirming the model
license and record the engine SHA256 with each benchmark run.

The primary DeepStream configuration also requires the external YOLO parser.
Build it locally with `scripts/build_custom_yolo_parser.sh`; its source checkout
and generated shared library are intentionally excluded from this repository.
