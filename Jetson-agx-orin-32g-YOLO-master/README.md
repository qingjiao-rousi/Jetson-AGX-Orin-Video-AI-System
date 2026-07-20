# Jetson AGX Orin 多路视频智能分析系统

本项目运行在 Jetson AGX Orin + DeepStream 7.1 上，使用一个 DeepStream Pipeline 处理八路本地 MP4 或 RTSP 视频。主模型负责目标检测和跟踪，专用模型通过场景路由按需执行，结果统一写入事件 JSONL 和输出视频。

## 当前能力

| 场景 | 摄像头 | 处理链路 |
|---|---|---|
| 车辆出入口 | 1 | YOLOv8 + tracker → vehicle → 车牌检测 → OCR |
| 生产区 | 2、3、4、6 | YOLOv8 + tracker → 安全帽、姿态、烟雾/火灾 |
| 普通区 | 5 | YOLOv8 + tracker → 人员/车辆统计、烟雾/火灾 |
| 仓库区 | 7、8 | YOLOv8 + tracker → 区域、越线、人车关系、烟雾/火灾 |

当前正式配置为：

```text
configs/app/app_multifile_8.yaml
```

配置固定为：

```yaml
source_count: 8
batch_size: 8
```

## 模型

| 模型 | 文件 | 用途 |
|---|---|---|
| 主模型 | `models/yolov8s.engine` | 人员、车辆等基础检测 |
| 安全帽 | `models/ppe_yolov8n_dynamic_fp16.engine` | `person → hardhat/no-hardhat` |
| 姿态 | `models/v8_n_pose_fp16.engine` | 生产区人员关键点 |
| 车牌检测 | `models/plate_detector_fastalpr.engine` | 车辆 ROI 内车牌检测 |
| 车牌 OCR | `models/plate_ocr_fastalpr-dynamic.engine` | 车牌字符识别 |
| 烟雾/火灾 | `models/fire_smoke_best_fp16.engine` | 整帧 fire/smoke 检测 |

专用任务不复制主 Pipeline，而是经过：

```text
DeepStream primary + tracker
→ C++ probe 逐路提取 metadata
→ scene/capability 路由
→ FrameStore 获取逐路 RGBA 帧
→ 有界 worker 执行专用 TensorRT engine
→ EventWriter 输出事件
```

## 环境

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master
source scripts/env.sh
```

`scripts/env.sh` 会配置 CUDA、TensorRT、DeepStream 和 PyTorch cuDSS 库路径。

检查环境：

```bash
scripts/check_env.sh
```

## 八路 MP4 验证

正式八路配置默认使用 `../video/1.mp4` 到 `../video/8.mp4`：

```bash
rm -rf outputs/multifile_inproc

ALL_CLASSES=1 \
OUTPUT_SINK=file \
scripts/run_multistream.sh \
  configs/app/app_multifile_8.yaml \
  outputs/multifile_inproc
```

查看模型初始化、路由和错误：

```bash
rg -n "routed model task|TensorRT backend initialized|task failed|ERROR" \
  outputs/multifile_inproc/app.log
```

统计事件和涉及的路数：

```bash
python3 - <<'PY'
import json
from collections import Counter

events = Counter()
streams = Counter()
with open("outputs/multifile_inproc/events.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        events[item["event_type"]] += 1
        streams[item["stream_id"]] += 1

print("events:", events)
print("streams:", streams)
PY
```

主要事件包括：

```text
helmet_violation
pose_observation
vehicle_pass
fire_smoke_detection
zone_observation
line_crossing
person_vehicle_relation
scene_statistics
```

## 单路模型验证

安全帽：

```bash
rm -rf outputs/helmet_mp4_test
OUTPUT_SINK=file scripts/run_multistream.sh \
  configs/app/app_helmet_mp4_test.yaml \
  outputs/helmet_mp4_test
```

车辆出入口：

```bash
rm -rf outputs/vehicle_gate_mp4_test
ALL_CLASSES=1 OUTPUT_SINK=file scripts/run_multistream.sh \
  configs/app/app_vehicle_gate_mp4_test.yaml \
  outputs/vehicle_gate_mp4_test
```

## 输出

每次运行目录通常包含：

```text
app.log              运行日志
results.jsonl        主模型逐帧结果
events.jsonl         业务事件
runtime_metrics.jsonl GPU、FPS、队列和 probe 指标
output.mp4           OSD 输出视频（文件 sink 时）
```

车牌事件中包含车牌文本和车牌框；安全帽、姿态、烟雾/火灾事件包含对应的置信度和坐标信息。

## 生成模型 engine

标准 Ultralytics 权重导出 ONNX：

```bash
MPLCONFIGDIR=/tmp/matplotlib \
python3 scripts/export_pt_to_onnx.py models/fire_smoke_best.pt
```

`v8_n_pose.pt` 来自 `tmp/YOLOv8-pose-master` 自定义源码，应使用：

```bash
MPLCONFIGDIR=/tmp/matplotlib \
python3 scripts/export_custom_pose_onnx.py \
  models/v8_n_pose.pt \
  --output models/v8_n_pose.onnx
```

ONNX 转 TensorRT：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/fire_smoke_best.onnx \
  --saveEngine=models/fire_smoke_best_fp16.engine \
  --fp16 --memPoolSize=workspace:4096 \
  --builderOptimizationLevel=3
```

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/v8_n_pose.onnx \
  --saveEngine=models/v8_n_pose_fp16.engine \
  --fp16 --memPoolSize=workspace:4096 \
  --builderOptimizationLevel=3
```

engine 应在目标 Jetson 设备上生成和验证，不能直接跨不同 GPU 架构复用。

## 测试与辅助脚本

运行单元测试（环境安装 pytest 后）：

```bash
python3 -m pytest -q tests/unit
```

正式主入口：

```text
scripts/run_multistream.sh       指定 YAML 的统一运行入口
configs/app/app_multifile_8.yaml 正式八路 MP4 配置
```

RTSP 和 MediaMTX 模拟入口：

```text
scripts/simulate_cameras.sh
scripts/run_rtsp_inproc.sh
scripts/run_production_service.sh
```

person analytics、批量报告和 Web 看板脚本仍保留作为离线分析/兼容验收工具，不属于专用模型主推理链路。

## 当前验收重点

核心模型和场景路由已经接入，剩余工作主要是工程验收：

- fire/smoke 置信度和连续帧去重调优；
- 车牌在八路并发下的稳定识别；
- 第 7、8 路仓库区域和越线坐标校准；
- 长时间运行、GPU 利用率和处理 FPS 验证；
- 根据实际摄像头画面调整各场景阈值。

