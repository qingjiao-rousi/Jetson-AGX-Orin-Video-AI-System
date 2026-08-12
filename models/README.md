# Model Assets

This repository tracks labels and model-side configuration only. It does not
include ONNX files, TensorRT engines, calibration images, weights, or videos.

Prepare the following local assets before running the full pipeline:

| Asset | Expected local path | Notes |
| --- | --- | --- |
| Primary YOLO ONNX | `export_yolov8_ds/yolov8s.onnx` | Build locally with the included DeepStream-Yolo-derived exporter; source for the current primary INT8 engine build. |
| Primary FP16 engine | `models/fp16/yolov8s.engine` | Build with `scripts/deploy/build_fp16_engines.sh --primary-only`. |
| Primary INT8 candidate engine | `models/int8/yolov8s_coco_train504.engine` | Current validated candidate; build with `scripts/deploy/build_primary_detector_int8.py` and the independent COCO train504 calibration set. |
| Specialist FP16 engines | `models/fp16/` | Build with `scripts/deploy/build_fp16_engines.sh --specialists-only`. |
| Candidate calibration images | `calibration/coco_train504/images/` | Independent COCO train2017 calibration set used by the current INT8 candidate. |
| Local videos | `video/1.mp4` through `video/8.mp4` | Required by the bundled multi-file examples. |
| COCO val2017 (optional) | `datasets/coco/` | Public, local-only labeled evaluation data; see `docs/coco_fp16_int8_evaluation.md`. |

TensorRT engines are specific to the target Jetson software and hardware
environment. Build them on the deployment device after confirming the model
license and record the engine SHA256 with each benchmark run.

The primary DeepStream configuration also requires the external YOLO parser.
Build it locally with `scripts/deploy/build_custom_yolo_parser.sh`; its source checkout
and generated shared library are intentionally excluded from this repository.

## Upstream YOLO export

`export_yolov8_ds/export_yoloV8.py` is adapted from upstream
[DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)
`utils/export_yoloV8.py` at commit `93aedb656a47b141ecbea99c407b002262287cfe`.
It uses the upstream MIT license retained at
`LICENSES/DeepStream-Yolo-93aedb656a47b141ecbea99c407b002262287cfe.txt`, not
this repository's Apache-2.0 license. The local modification changes only the
`--weights` help text. Record the upstream revision together with the
model-weight source, command, and output SHA256. The resulting ONNX and
weights remain local-only artifacts.
