# codex.md — Jetson AGX Orin + DeepStream 多路视频实时检测系统

## 项目概述

基于 NVIDIA Jetson AGX Orin + DeepStream SDK 的边缘端多路视频实时目标检测系统。
**Python 主控 + 少量 C++/CUDA 插件**，分层架构（domain → application → infrastructure → optimization）。
目标：6–12 路 1080P RTSP 实时检测 + 跟踪 + JSONL/MQTT/Kafka 结构化输出。

## 环境

### 目标机（Jetson AGX Orin）

- Ubuntu + JetPack 6.2.2 + DeepStream 7.1 + CUDA 12.x + TensorRT 10.x
- 所有 DeepStream/CUDA 相关代码**只能在此验证
- Python 3.10+，虚拟环境 `.venv/`
- Python 需要能 `import pyds` 和 `import gi`（GStreamer Python bindings）

### 环境自适应设计

项目代码已做离线兼容：当 `gi`/`pyds` 不可导入时，`GStreamerRuntimeFactory` 会将 `available` 设为 `False`，pipeline 构建走 dry-run 模式，不会崩溃。详见 `src/app/infrastructure/pipeline/builder.py:33-95`。

## 目录结构（可能会发生文件增删）

```
src/app/                    — 主业务代码
  main.py                   — 入口，只负责启动
  bootstrap.py              — 依赖装配（创建所有对象并互联）
  settings.py               — 统一配置定义（dataclass），所有模块从此读配置
  domain/entities.py        — 核心数据模型：BoundingBox, Detection, Track, FrameResult, StreamStats
  application/orchestrator.py — 总调度器：启停、帧结果分发、异常处理
  infrastructure/
    pipeline/builder.py     — DeepStream pipeline 构建（节点定义、链接、probe）
    pipeline/manager.py     — pipeline 生命周期管理（start/stop/restart/bus watch）
    pipeline/source_factory.py — 视频源规范化（RTSP/文件 → SourceBranchSpec）
    pipeline/probes.py      — probe 回调注册与分发
    inference/meta_parser.py — 从 NvDsBatchMeta 提取 Detection/Track → FrameResult
    output/json_writer.py   — 线程安全的 JSONL 追加写入
    monitoring/gpu_monitor.py — GPU/FPS 监控（tegrastats）
    web/dashboard.py        — HTTP API + 静态页面（ThreadingHTTPServer, daemon 线程）
  optimization/
    fps_controller.py       — 自适应丢帧：GPU>85% 或 queue>80% → 阶梯概率丢帧
    backpressure_controller.py — 背压保护：produce/consume 时间戳对账，队列深度监控
    strategy_advisor.py     — 优化建议生成
  adapters/config_loader.py — YAML → AppSettings dataclass
  shared/logger.py          — 日志初始化 + 内存环形缓冲（供 Web 查询）

configs/
  app/app.yaml              — 业务配置（源数量、模型路径、输出开关）
  app/streams.yaml          — 视频源列表
  app/output.yaml           — 输出目标配置
  deepstream/               — DeepStream 插件配置（infer/tracker/streammux）
  logging/logging.yaml      — 日志配置

custom_libs/                — C++ 自定义插件（DeepStream parser .so, CUDA kernel）
scripts/                    — 辅助脚本（download_models.sh, preview_web.py）
models/                     — 模型产物（.onnx, .engine, labels.txt）
tests/unit/                 — 纯 Python 单元测试（7 个文件，不依赖 DeepStream）
```

## 常用命令

### Python 环境

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install pyyaml
```

### 测试

```bash
# 跑全部单元测试
python -m pytest tests/unit/ -v

# 跑单个测试文件
python -m pytest tests/unit/test_pipeline_builder.py -v

# 跑单个测试函数
python -m pytest tests/unit/test_pipeline_builder.py::test_build_minimal_pipeline -v
```

### Web 预览

```bash
python scripts/preview_web.py
```

### 模型准备（在 Jetson 上执行）

```bash
# 一键下载+导出+构建（约 30 分钟）
bash scripts/download_models.sh

# 或分步：
# Step 1: 下载 PyTorch 权重
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O models/yolov8n.pt

# Step 2: 导出 ONNX
python3 -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt').export(format='onnx', opset=17, imgsz=640)"

# Step 3: 构建 TensorRT Engine（FP16）
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolov8n.onnx \
    --saveEngine=models/yolov8n.engine \
    --fp16 --workspace=2048 \
    --optShapes=input:6x3x640x640 \
    --minShapes=input:1x3x640x640 \
    --maxShapes=input:6x3x640x640
```

### 运行主程序（在 Jetson 上）

```bash
# 默认配置启动
python3 src/app/main.py

# 指定配置文件
python3 src/app/main.py --config configs/app/app.yaml

# 单路本地视频调试（修改 app.yaml: source_count: 1 + streams.yaml 用 file:// 源）
```

### DeepStream 环境验证（在 Jetson 上）

```bash
deepstream-app --version              # DeepStream 是否安装
python3 -c "import pyds; print('ok')"  # pyds 是否可用
python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print('ok')"  # GStreamer 是否可用
gst-launch-1.0 --version              # gst-launch 是否可用
tegrastats --interval 1000            # GPU 监控是否可用
```

## 代码规范

### 命名

- 文件：`snake_case`（`meta_parser.py`, `json_writer.py`）
- 类：`PascalCase`（`PipelineManager`, `FpsController`）
- 函数/变量：`snake_case`
- 配置类：`XxxSettings`（`AppSettings`, `DeepStreamSettings`）

### 分层规则（架构底线，不可违反）

1. `domain/` — 纯 `@dataclass(frozen=True)` 数据模型，**不 import 任何外部库**，不依赖 infrastructure
2. `application/` — 流程编排，**只调用接口**，不直接操作 `pyds`/`Gst`
3. `infrastructure/` — **唯一能 import `pyds`/`gi`/DeepStream 的地方**
4. `optimization/` — 策略决策，不直接改写底层对象，通过服务接口驱动
5. `shared/` — 只放真正通用的工具（日志、异常），不塞业务逻辑
6. `adapters/` — 边界适配（配置加载、事件总线），解耦层间依赖

### 配置

- 所有配置通过 `settings.py` 中的 dataclass 统一读取
- 不允许在业务代码中直接 `open()` YAML 文件
- 配置项用 `snake_case`，YAML 文件中用 `snake_case`
- 新增配置 → 先在 `settings.py` 加字段，再在 `app.yaml` 加值，最后在 `config_loader.py` 加映射

### 数据流

- 模块间数据传递统一使用 `domain/entities.py` 中定义的数据类
- `FrameResult` 是核心流转对象：从 probe 回调 → meta_parser → orchestrator → json_writer
- 输出模块统一消费 domain 对象，**不耦合 DeepStream 原始元数据**（`NvDsBatchMeta` 等）

## 关键设计决策

### Probe → 领域对象转换链路

```
Pipeline probe 回调 (builder.py:_on_probe_buffer)
  → _extract_nvds_batch_meta (取出 NvDsBatchMeta)
  → _batch_meta_to_payload (遍历 GLib list → dict)
  → ProbeRegistry.emit_probe_payload
  → Orchestrator.on_frame_result
  → MetaParser.parse (dict → FrameResult dataclass)
  → JsonWriter.write (dataclass → asdict → json.dumps)
```

### 丢帧策略（FpsController）

- 阶梯式调整：GPU>85% 或 queue>80% → `drop_rate += 0.15`；GPU<60% 且 queue<40% → `drop_rate -= 0.05`
- **概率丢帧**而非固定模式，避免与摄像头帧率产生同步共振
- `drop_rate` 上限 90%，防止 pipeline 完全停滞
- 离线模式下 `_read_gpu()` 返回 50%（安全默认值，不触发丢帧）

### 背压保护（BackpressureController）

- 生产者每帧调用 `observe()`，消费者写完调用 `mark_consumed()`
- 队列深度 ≥ 75% max_queue_size → `backpressure_active = True`
- 队列深度 ≤ 30% → 解除
- 当前版本 `orchestrator.py:50-57` 中 produce/consume 仍在同一线程（未异步化）

## 调试指南

### 排查流程（按顺序）

1. **看日志** — `outputs/logs/app.log`，确认错误类型和位置
2. **检查配置** — `configs/app/app.yaml`，重点看路径、source_count、enable_*
3. **验证环境** — DeepStream/CUDA/TensorRT 版本是否匹配
4. **最小复现** — 用单路本地视频（`file://`）替代多路 RTSP

### 常见问题


| 现象                                            | 原因                                      | 排查                                                          |
| ----------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------- |
| `ImportError: No module named pyds`             | 不在 Jetson 上                            | 正常，离线兼容已处理；如需真实推理必须在 Jetson 上跑          |
| `pipeline builder: model engine file not found` | `models/*.engine` 不存在                  | 在 Jetson 上执行`scripts/download_models.sh`                  |
| pipeline 卡死不推流                             | streammux batch-size 与实际输入路数不匹配 | 检查`source_count` == 启用源数 == `batch_size`                |
| JSONL 输出为空                                  | meta_parser 解析不到数据                  | 检查 probe 是否正确附加、custom_libs/*.so 是否正确加载        |
| 单帧异常导致崩溃                                | meta_parser 缺异常处理                    | 当前`meta_parser.py` 无 try/except 包裹（已知问题 #6）        |
| Web 页打不开                                    | `enable_web: false`                       | 改`configs/app/app.yaml` 中 `enable_web: true`，端口默认 8080 |
| `source_count` 与启用的源数不匹配               | 配置不一致                                | `builder.py:_validate_source_count` 会抛 ValueError           |
| MQTT 报错                                       | `enable_mqtt: true` 但未配置 host         | `settings.py:100-101` 会抛 ValueError                         |

### 加调试日志

```python
# 在代码中加日志（不要在 probe 热路径打太多）
import logging
logging.getLogger("deepstream.probe").debug("frame %s: %d detections", stream_id, len(detections))

# 通过 Web API 查看运行时日志（需 enable_web: true）
curl http://127.0.0.1:8080/api/logs?limit=50
curl http://127.0.0.1:8080/api/status
curl http://127.0.0.1:8080/api/debug
```

## 当前状态

- **阶段**：第一阶段（跑通主链路），代码框架已完整，待上板验证
- **已完成**：目录结构、配置体系、数据模型、pipeline 构建（含离线兼容）、JSONL 输出、FPS 控制器、背压控制器、Web Dashboard、单元测试（7 个文件）
- **待完成（上板前必须）**：
  1. 模型文件获取（`models/yolov8n.engine` 不存在）
  2. `meta_parser.py` 加 try/except 异常保护
  3. `json_writer.py` 加序列化异常保护
  4. `orchestrator.py` 的 `handle_error()` 改为有序重启而非抛异常
- **待完成（上板后）**：多模型 SGIE 架构、JSON 写出异步化（独立消费线程）、GPU 监控独立线程
- **详细待办**：见 `项目优化路线图.md` 和 `项目问题解决方案.md`

## 相关文档索引

- `AGX.md` — 项目最终定义、系统架构、核心设计思路、C++/CUDA 模块设计
- `项目代码架构设计.md` — 完整的分层架构说明和命名规范
- `项目问题解决方案.md` — #1~#7 号问题的详细改动方案（多模型、丢帧、多线程、异常处理、内存对齐）
- `项目开发前置清单.md` — 环境准备、模型资产、配置冻结 checklist
- `项目优化路线图.md` — 分阶段进度追踪
- `最小可运行版本指南.md` — 从零到跑通的步骤指南
- `知识体系指南.md` — 知识点索引和学习路线
