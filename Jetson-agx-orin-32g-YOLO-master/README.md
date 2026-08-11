# Jetson DeepStream 多路视频智能分析

运行在 Jetson AGX Orin 32GB 上的 DeepStream + TensorRT + YOLOv8 多路视频分析系统。主 Pipeline 负责解码、batch 推理、tracker、OSD 和输出；Python 业务层按场景和 capability 调度安全帽、姿态、烟火、车牌检测与 OCR 任务。

## 已验证能力

- 8 路本地 MP4 输入，单 DeepStream Pipeline，`nvstreammux batch_size=8`。
- file、fake 和 RTSP/RTMP sink 代码路径；8 路 MP4 INT8 完整输出已在 Jetson 实测。
- C++ `NvDsBatchMeta` probe + Python fallback；结果输出为逐帧 JSONL 和事件 JSONL。
- 安全帽、姿态、火焰烟雾、车牌检测使用 INT8 engine；车牌 OCR 当前保持 FP16。
- `tegrastats` 指标、队列状态、GPU/温度/RAM/功耗记录，以及 systemd/日志轮转模板。

当前没有声称：真实摄像头长时间稳定性、模型 mAP/Recall、FP16 与 INT8 的公平对比，或任何未完成实验支持的 P95 延迟数值。

## 系统结构

```text
MP4 / RTSP
   -> hardware decode -> nvstreammux(batch=8) -> YOLOv8 INT8 -> tracker
   -> C++ metadata probe -> Python FrameResult
   -> capability routing -> bounded TensorRT workers -> JSONL events
   -> OSD -> hardware H.264 encoder -> MP4 / RTMP / RTSP
```

关键代码边界：

- `src/app/infrastructure/pipeline/`：GStreamer/DeepStream pipeline、source、bus、probe。
- `custom_libs/probe_handler/`：C++ batch metadata parser。
- `src/app/application/`：业务路由、专用 TensorRT worker、场景分析。
- `src/app/infrastructure/monitoring/`：`tegrastats` 与 runtime metrics。
- `scripts/`：环境检查、engine 构建、运行、验收和汇总。

## 环境

已验证环境：Jetson AGX Orin 32GB、JetPack 6.2.1、TensorRT 10.7.0、DeepStream 7.1、Python 3.10。engine 必须在目标 Jetson 上构建，不能假设跨 GPU 架构兼容。

```bash
source scripts/env.sh
scripts/check_env.sh
```

依赖安装脚本会安装系统包并构建 `pyds`，需要 Jetson、sudo 和网络：

```bash
scripts/install_jetson_deps.sh
```

## 模型与配置

engine、ONNX、权重、校准图片和视频不属于公开代码提交。请根据模型许可证准备这些文件，并在本机生成 engine。

主模型 A/B 基准配置：[configs/app/app_multifile_8_primary_int8.yaml](configs/app/app_multifile_8_primary_int8.yaml)。它只将主 YOLO 替换为 INT8，安全帽、姿态、烟火、车牌检测和 OCR 继续使用 FP16。`app_multifile_8_int8.yaml` 保留为后续全辅助模型 INT8 的场景配置，不用于当前主模型对照。

主模型 INT8 校准构建示例：

```bash
python3 scripts/build_yolov8s_int8.py \
  --onnx models/yolov8s.onnx \
  --images calibration/yolov8s \
  --batch-size 8 \
  --cache models/int8/yolov8s_int8_calibration.cache \
  --engine models/int8/yolov8s_int8.engine
```

该脚本使用 TensorRT `IInt8EntropyCalibrator2`，属于隐式 PTQ；警告中的缺失 scale 可能使部分层回退到非 INT8，不能把它表述为“全图完全 INT8”。`quantize_yolov8s_qdq.py` 是显式 Q/DQ 实验脚本，不是默认运行链路。

## 运行 8 路主模型 INT8 MP4

输入配置中的视频路径是项目目录外的 `../video/1.mp4` 到 `../video/8.mp4`。准备好本机 engine 和视频后：

```bash
OUTPUT_SINK=file RUN_SECONDS=0 ENABLE_TEGRASTATS=1 \
scripts/run_multistream.sh \
  configs/app/app_multifile_8_primary_int8.yaml \
  outputs/primary_int8_multifile_8streams
```

输出目录包含 `output.mp4`、`results.jsonl`、`events.jsonl`、`runtime_metrics.jsonl` 和 `app.log`。

```bash
python3 scripts/summarize_precision_run.py \
  --run primary_int8=outputs/primary_int8_multifile_8streams \
  --warmup-samples 5 \
  --output outputs/int8_summary.json
```

`OUTPUT_SINK=fake` 可排除编码/写盘，用于压力基线；`file` 用于验证完整输出链路，两者不能直接当作同一性能指标比较。

## RTSP 与运维

本地 MP4 可通过 MediaMTX 模拟 RTSP：

```bash
scripts/serve_mp4_as_rtsp.py ../video --limit 8
scripts/run_rtsp_inproc.sh outputs/rtsp_run
```

服务模板位于 `deploy/`，使用前必须替换其中的 `@PROJECT_ROOT@`、用户、视频目录和日志目录占位符。不要把真实 RTSP 地址、密码或证书提交到仓库。

## 测试

项目测试使用 Python 标准库 `unittest`，不依赖 pytest：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/unit -p 'test_*.py' -v
```

测试覆盖 metadata、probe、路由、pipeline blueprint、状态管理、JSONL、RTSP simulator、质量汇总和 Web API。Jetson GStreamer/CUDA 相关测试应在目标硬件上执行。

## 实测记录

修复 batch metadata 统计后，8 路本地 MP4 INT8 正常文件输出记录于 [projectMd/精度对比实验报告.md](projectMd/精度对比实验报告.md)：14,222 条结果帧、估算丢帧 0%、聚合处理 FPS 80.381。该数据只代表指定硬件、软件、视频和配置，不能替代 mAP 或公平的 FP16/INT8 对比。性能实验的统一口径、P50/P95 定义和 36 组矩阵执行方式见 [../docs/benchmark.md](../docs/benchmark.md)。

## 已知限制与路线

- 当前 OCR 仍为 FP16，INT8 是混合运行链路而非所有模型纯 INT8。
- 缺少固定标注集，因此暂无 mAP、Precision、Recall、误检/漏检结果。
- 已埋点主推理前探针至 JSON 成功写入的 P50/P95；仍需按统一矩阵完成 1/4/8 路 FP16/INT8 实测、真实 RTSP 长稳和故障注入。
- 需要将 benchmark 配置、环境信息、engine SHA256 和质量结果作为脱敏实验记录保存。
