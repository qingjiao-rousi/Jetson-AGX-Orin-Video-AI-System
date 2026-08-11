# Model Assets

This repository tracks labels and model-side configuration only. It does not
include ONNX files, TensorRT engines, calibration images, weights, or videos.

Prepare the following local assets before running the full pipeline:

| Asset | Expected local path | Notes |
| --- | --- | --- |
| Primary YOLO ONNX | `models/yolov8s.onnx` | Source for FP16 and INT8 engine builds. |
| Primary FP16 engine | `models/fp16/yolov8s.engine` | Build with `scripts/build_fp16_engines_trt107.sh --primary-only`. |
| Primary INT8 engine | `models/int8/yolov8s_int8.engine` | Build with `scripts/build_yolov8s_int8.py`. |
| Specialist FP16 engines | `models/fp16/` | Build with `scripts/build_fp16_engines_trt107.sh --specialists-only`. |
| Calibration images | `calibration/yolov8s/` | Representative images for primary INT8 calibration. |
| Local videos | `video/1.mp4` through `video/8.mp4` | Required by the bundled multi-file examples. |

TensorRT engines are specific to the target Jetson software and hardware
environment. Build them on the deployment device after confirming the model
license and record the engine SHA256 with each benchmark run.

The primary DeepStream configuration also requires the external YOLO parser.
Build it locally with `scripts/build_custom_yolo_parser.sh`; its source checkout
and generated shared library are intentionally excluded from this repository.
