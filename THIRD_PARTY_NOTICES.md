# Third-Party Notices and Asset Boundaries

This repository's Apache-2.0 [LICENSE](LICENSE) applies only to original
source code, configuration, documentation, and diagrams authored for this
repository. It does not grant rights to any third-party software, SDK, model,
weight, engine, dataset, media file, or trademark.

The following components are obtained separately and remain subject to their
respective upstream terms. This is a notice of dependency, not a relicensing
or a complete legal inventory. Before redistributing or using this project in
a commercial setting, review the current upstream terms yourself.

| Component | How this repository uses it | License / terms source | Distributed here |
| --- | --- | --- | --- |
| NVIDIA JetPack, CUDA, TensorRT, DeepStream and `pyds` | Target runtime and Python bindings | [NVIDIA Software License Agreement](https://developer.nvidia.com/jetpack-sdk-archive) and the installed SDK notices | No |
| DeepStream-Yolo | Local source checkout used to build the custom YOLO parser; adapted `export_yoloV8.py` | [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo), commit `93aedb656a47b141ecbea99c407b002262287cfe` | `export_yolov8_ds/export_yoloV8.py` is derived from `utils/export_yoloV8.py` and uses the upstream MIT license retained at `LICENSES/DeepStream-Yolo-93aedb656a47b141ecbea99c407b002262287cfe.txt`. Local change: `--weights` help text only. Generated parser `.so` is ignored |
| Ultralytics YOLO / `ultralytics` | Runtime dependency of the adapted upstream ONNX exporter | [Ultralytics licensing](https://www.ultralytics.com/license) | No package, weights, or Ultralytics source vendored |
| COCO | Optional INT8 calibration and detector evaluation data | [COCO terms of use](https://cocodataset.org/#termsofuse) | No images, annotations, or predictions |
| MediaMTX and FFmpeg | Optional local RTSP simulation tools | Their respective upstream repositories and installed-package notices | No |

## Models, engines, and media

Model weights, ONNX files, TensorRT engines, calibration images, input videos,
and generated outputs are local-only artifacts and are intentionally excluded
by `.gitignore`. Their source, license, consent, privacy obligations, and
redistribution permissions must be verified by the operator. TensorRT engines
are also tied to the target hardware and software stack; build them locally on
the deployment Jetson.

Names such as NVIDIA, Jetson, TensorRT, DeepStream, YOLO, Ultralytics, COCO,
MediaMTX, and FFmpeg may be trademarks of their respective owners. Their use
in this repository is descriptive only and does not imply endorsement.

## Exporter provenance

The exporter provenance was verified against DeepStream-Yolo commit
`93aedb656a47b141ecbea99c407b002262287cfe`. The local file differs only in
the `--weights` help text; its functional implementation matches the upstream
file. The exporter imports Ultralytics APIs at runtime, so operators must also
review the current Ultralytics license before use or redistribution.
