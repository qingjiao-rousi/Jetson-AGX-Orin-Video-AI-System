# Jetson DeepStream 多路视频智能分析

运行在 Jetson AGX Orin 32GB 上的 DeepStream + TensorRT + YOLOv8 多路视频分析系统。主 Pipeline 负责解码、batch 推理、tracker、OSD 和输出；Python 业务层按场景和 capability 调度安全帽、姿态、烟火、车牌检测与 OCR 任务。

## 已验证能力

- 8 路本地 MP4 输入，单 DeepStream Pipeline，`nvstreammux batch_size=8`。
- file、fake 和 RTSP/RTMP sink 代码路径；8 路 MP4 主模型 FP16/INT8 完整输出已在 Jetson 实测。
- C++ `NvDsBatchMeta` probe + Python fallback；结果输出为逐帧 JSONL 和事件 JSONL。
- 安全帽、姿态、火焰烟雾、车牌检测与 OCR 由独立 TensorRT worker 按配置加载；当前主模型 A/B 基线中它们均保持 FP16、batch=1。
- `tegrastats` 指标、队列状态、GPU/温度/RAM/功耗记录，以及 systemd/日志轮转模板。

当前没有声称：真实摄像头长时间稳定性，或真实业务域的标注质量。COCO val2017 已完成主模型 FP16/INT8 标注评测；性能结论仅适用于 [benchmark.md](docs/benchmark.md) 记录的 Jetson、输入视频、功耗模式与软件版本。

## 系统结构

```text
MP4 / RTSP
   -> hardware decode -> nvstreammux(batch=1/4/8) -> primary YOLO (FP16/INT8) -> tracker
   -> C++ metadata probe -> Python FrameResult
   -> capability routing -> bounded TensorRT workers -> JSONL events
   -> OSD -> hardware H.264 encoder -> MP4 / RTMP / RTSP
```

关键代码边界：

- `src/app/infrastructure/pipeline/`：GStreamer/DeepStream pipeline、source、bus、probe。
- `custom_libs/probe_handler/`：C++ batch metadata parser。
- `src/app/application/`：业务路由、专用 TensorRT worker、场景分析。
- `src/app/infrastructure/monitoring/`：`tegrastats` 与 runtime metrics。
- `scripts/`：按部署、基准、评测、RTSP、工具和历史子系统分组；入口说明见 [scripts/README.md](scripts/README.md)。

## 环境

已验证环境：Jetson AGX Orin 32GB、JetPack 6.2.1、TensorRT 10.7.0、DeepStream 7.1、Python 3.10。engine 必须在目标 Jetson 上构建，不能假设跨 GPU 架构兼容。

```bash
source scripts/deploy/env.sh
scripts/deploy/check_env.sh
```

依赖安装脚本会安装系统包并构建 `pyds`，需要 Jetson、sudo 和网络：

```bash
scripts/deploy/install_jetson_deps.sh
```

## 模型与配置

engine、ONNX、权重、校准图片和视频不属于公开代码提交。请根据模型许可证准备这些文件，并在本机生成 engine。

模型资产清单与构建边界见 [models/README.md](models/README.md)，文档导航见 [docs/index.md](docs/index.md)。

主检测配置依赖 DeepStream YOLO 自定义解析器。该解析器是与 Jetson 环境绑定的本机构建产物，未纳入 Git；在目标设备上执行：

```bash
scripts/deploy/build_custom_yolo_parser.sh
```

脚本会打印上游源代码提交和 `.so` 的 SHA256，应将两者记录到实验记录中。若要固定上游版本，设置 `DEEPSTREAM_YOLO_REVISION=<commit-or-tag>` 后再执行。

主模型 A/B 基准配置：[configs/app/app_multifile_8_primary_int8.yaml](configs/app/app_multifile_8_primary_int8.yaml)。它只将主 YOLO 替换为通过 COCO train504 校准的 INT8 候选，安全帽、姿态、烟火、车牌检测和 OCR 继续使用 FP16。旧 `yolov8s_int8.engine` 仅保留作历史量化质量对照；`app_multifile_8_int8.yaml` 是后续全辅助模型 INT8 的场景配置，不用于当前主模型对照。

主模型 INT8 校准构建示例：

```bash
python3 scripts/deploy/build_primary_detector_int8.py \
  --onnx export_yolov8_ds/yolov8s.onnx \
  --images calibration/coco_train504/images \
  --batch-size 8 \
  --cache models/int8/yolov8s_coco_train504_calibration.cache \
  --engine models/int8/yolov8s_coco_train504.engine
```

## 运行 8 路主模型 INT8 MP4

输入配置使用仓库根目录下的 `video/1.mp4` 到 `video/8.mp4`。准备好本机 engine 和视频后：

```bash
OUTPUT_SINK=file RUN_SECONDS=0 ENABLE_TEGRASTATS=1 CONFIDENCE_THRESHOLD=0.15 \
scripts/deploy/run_multistream.sh \
  configs/app/app_multifile_8_primary_int8.yaml \
  outputs/primary_int8_coco_train504_8streams
```

输出目录包含 `output.mp4`、`results.jsonl`、`events.jsonl`、`runtime_metrics.jsonl` 和 `app.log`。

```bash
python3 scripts/benchmark/summarize_precision_run.py \
  --run primary_int8=outputs/primary_int8_coco_train504_8streams \
  --warmup-samples 5 \
  --output outputs/int8_summary.json
```

`OUTPUT_SINK=fake` 可排除编码/写盘，用于压力基线；`file` 用于验证完整输出链路，两者不能直接当作同一性能指标比较。

## RTSP 与运维

本地 MP4 可通过 MediaMTX 模拟 RTSP：

```bash
scripts/rtsp/serve_mp4_as_rtsp.py video --limit 8
scripts/rtsp/run_rtsp_inproc.sh outputs/rtsp_run
```

服务模板位于 `deploy/`，使用前必须替换其中的 `@PROJECT_ROOT@`、用户、视频目录和日志目录占位符。不要把真实 RTSP 地址、密码或证书提交到仓库。

## 测试

项目测试使用 Python 标准库 `unittest`，不依赖 pytest：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/unit -p 'test_*.py' -v
```

测试覆盖 metadata、probe、路由、pipeline blueprint、状态管理、JSONL、RTSP simulator、质量汇总和 Web API。Jetson GStreamer/CUDA 相关测试应在目标硬件上执行。

## 实测记录

已完成主 YOLO FP16/INT8 的 `1/4/8` 路、`fake/file`、每组 3 次重复的 36 组系统基线，以及 COCO val2017 的完整标注质量评测。最新候选还完成了 8 路 `fake/file`、每组 3 次的系统复测：在对应 FP16 对照下，吞吐提高约 `7.5%/6.3%`、E2E P50 降低约 `13.0%/11.9%`，但 FrameStore 丢弃仍是独立瓶颈。性能口径和结果见 [docs/benchmark.md](docs/benchmark.md)；COCO FP16/INT8 质量差异与部署结论见 [docs/coco_fp16_int8_evaluation.md](docs/coco_fp16_int8_evaluation.md)。系统吞吐结果不能替代 mAP，反之亦然。

## 已知限制与路线

- 已埋点主推理前探针至 JSON 成功写入的 P50/P95；已完成 COCO val2017 的主模型质量对比。后续优先补真实业务标注帧、RTSP 长稳和故障注入。
- 需要将 benchmark 配置、环境信息、engine SHA256 和质量结果作为脱敏实验记录保存。
