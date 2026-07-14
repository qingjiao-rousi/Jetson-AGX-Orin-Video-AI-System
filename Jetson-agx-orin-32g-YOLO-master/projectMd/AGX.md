# 一、项目最终定义

## 🎯 项目名称

基于 Jetson AGX Orin + DeepStream 的多路视频实时目标检测与边缘推理优化系统*

## 🎯 项目定位

> 面向安防/工业检测场景的边缘端多路视频实时分析系统，实现高并发视频接入、低延迟目标检测与结构化数据输出，并通过工程级优化显著提升推理吞吐能力。

## 🎯 核心目标

在资源受限的 Jetson AGX Orin 上实现：

* 6–12 路 1080P RTSP 视频接入
* 实时目标检测 + 跟踪（待优化）
* 稳定低延迟输出结构化结果
* 最大化 GPU 利用率与吞吐能力

## ✅ 当前落地验收口径

当前阶段没有真实 RTSP 摄像头资源，因此项目验收采用“本地 MP4 模拟 RTSP 摄像头”的方式作为最终当前版本：

```text
本地 MP4 文件
    ↓
FFmpeg 循环推流
    ↓
MediaMTX 提供 8 路 RTSP
    ↓
单 Python/DeepStream 进程内多路 pipeline
    ↓
nvstreammux batch 推理
    ↓
person 检测 / tracker / ROI / 越线 / runtime metrics
    ↓
JSONL、summary、quality、UI 展示
```

这与真实摄像头接入的主链路一致，后续真实落地时主要替换输入源：

```text
rtsp://127.0.0.1:8555/stream1
```

替换为真实摄像头：

```text
rtsp://camera-ip/path
```

当前一条命令验收入口：

```bash
RUN_SECONDS=40 START_UI=1 CHECK_RECOVERY=1 scripts/run_production_acceptance.sh
```

生产验收默认不生成 2x4 合并视频，而是生成每路独立的 OSD 推理视频：

```text
outputs/production_acceptance_latest/individual/stream_01/stream_01_osd.mp4
...
outputs/production_acceptance_latest/individual/stream_08/stream_08_osd.mp4
```

UI 高级实时调试区会按路分别播放这些视频，用于观察检测框、`track_id`、误检和漏检。若需要合并预览，可显式设置 `ENABLE_TILED_OUTPUT=1 OUTPUT_SINK=file`。

当前质量规则由 `scripts/check_rtsp_inproc_outputs.py` 固化，并输出到：

```text
outputs/production_acceptance_latest/rtsp_quality.json
```

主要判定：

* 8 路 RTSP 模拟源全部 online
* 单 pipeline 正常退出或按运行时长结束
* `results.jsonl` 非空且无坏 JSON
* 8 路 stream 均有帧输出
* runtime metrics 正常写入
* source health 无异常 stale
* run.log 无 fatal 错误
* tegrastats 能提供真实 Jetson GPU / 内存 / 温度 / 功耗指标

# 二、系统整体架构

```
RTSP多路视频流
        ↓
GStreamer + NVDEC（硬件解码）
        ↓
nvstreammux（多路批处理/帧同步）
        ↓
TensorRT YOLOv8n（FP16 / INT8推理）
        ↓
nvtracker（目标跟踪）
        ↓
工程级优化模块（关键）
        ↓
结构化输出（JSON / MQTT / Kafka）
```

# 三、核心技术栈

## 🧱 基础框架

* NVIDIA DeepStream SDK
* GStreamer Pipeline

## ⚡ 推理加速

* TensorRT（FP16 / INT8）
* ONNX Runtime（辅助转换）

## 🤖 模型

* YOLOv8n（主检测模型）
* 可选 YOLOv8n-seg（扩展）

## 🎥 视频处理

* NVDEC 硬件解码
* nvstreammux 多路批处理

## 📦 多目标处理

* nvtracker（IOU / NvDCF）

## 📡 数据输出

* JSON结构化输出
* MQTT / Kafka（可选）

## 🔧 高性能加速语言

* **Python** — 配置管理、流程编排、Web Dashboard、非热路径逻辑
* **C++** — DeepStream custom parser、probe 热路径回调、共享内存 IPC
* **CUDA C** — 预处理/后处理 kernel（ROI 裁剪、NMS、自定义 OSD 叠加）
* **编译产物** — `.so` 动态库由 DeepStream/GStreamer 运行时加载

# 四、系统核心设计思路

这个项目的本质不是“AI模型”，而是：

> **一个围绕 GPU 资源调度的多路实时视频推理系统**

核心设计原则：

### ✔️1. 推理优先，而不是模型复杂度优先

* 用 YOLOv8n 保证基础性能
* 不追求大模型

### ✔️2. 系统瓶颈优先优化

真正瓶颈在：

* 解码
* batching
* GPU调度
* frame pipeline

### ✔️3. 工程稳定性优先

* 防止 queue overflow
* 防止 pipeline 崩溃
* 控制 latency 波动

# 五、核心功能模块

## 🟢 1. 多路视频接入模块

* RTSP流接入
* NVDEC硬件解码
* GStreamer zero-copy pipeline

## 🟢 2. 多路批处理模块

* nvstreammux合并多路frame
* batch size固定调优
* timeout控制延迟

## 🟢 3. 实时目标检测

* YOLOv8n TensorRT推理
* FP16 / INT8加速

## 🟢 4. 多目标跟踪

* nvtracker（IOU / NvDCF）
* ID tracking + trajectory

## 🟢 5. 工程优化控制模块

### ⭐（1）GPU负载自适应帧率控制

根据GPU利用率动态调整：

* GPU > 85% → 降低输入FPS / drop frame
* GPU < 50% → 恢复FPS
* queue overflow → 强制丢帧

### ⭐（2）ROI裁剪优化

* 仅对关键区域推理
* 非ROI区域跳过检测

### ⭐（3）Smart Frame Skipping

* 目标稳定 → skip N帧
* 目标变化 → full inference

### ⭐（4）batch size tuning

* 根据吞吐与延迟调优 batch size
* 固定最优配置（工程调参结果）

### ⭐（5）backpressure控制机制

* 队列长度监控
* 防止 pipeline 堆积
* 丢帧保护机制

## 🟢 6. 结构化输出模块

* bbox + class + confidence
* tracking ID
* timestamp
* 输出 JSON / MQTT / Kafka

# 六、优化体系

这是项目的“核心价值部分”。

## ⚡ 1. 推理优化（Inference Optimization）

### ✔️TensorRT FP16 / INT8

* FP16默认加速
* INT8量化（calibration dataset）

👉 效果：

* FPS提升 30%–80%

### ✔️模型裁剪（可选增强）

* channel pruning
* layer pruning

👉 效果：

* 模型体积减少 20–40%
* 延迟下降

### ✔️知识蒸馏（高级优化）

* YOLOv8l → YOLOv8n

# ⚡ 2. 视频流优化（Pipeline Optimization）

### ✔️nvstreammux调优

* batch size tuning（6–12路）
* timeout控制（20–40ms）

### ✔️动态分辨率

* 1080p → 640×640推理
* ROI区域高清保留

### ✔️frame skipping策略

* 稳定场景跳帧
* 动态减少计算量

# ⚡ 3. 系统级优化（System Optimization）

### ✔️GPU负载控制

* 利用率驱动帧率调节

### ✔️backpressure机制

* queue length监控
* 自动丢帧防止堆积

### ✔️pipeline稳定性设计

* 防止内存爆炸
* 控制延迟抖动

# 七、C++ 高性能加速模块（核心竞争力层）

> **设计原则**：Python 管编排、配置、Web 展示和非热路径逻辑；C++/CUDA 管 DeepStream 热路径、推理输出解析、元数据提取和底层性能优化。
> **优化原则**：先做“功能必需 + 真实收益大”的 C++ 模块，再做“高阶底层优化模块”，避免第一阶段过度设计。

## 🔴 模块一：YOLO Custom Parser `.so`（功能必需 + 性能关键）

### 是什么

DeepStream 的 `nvinfer` 插件并不知道 YOLOv8 的输出格式，因此必须通过 **C++ 自定义解析器**，把推理引擎输出的 raw tensor 解码成 `NvDsInferObjectDetectionInfo` 列表。

这一模块本质上属于：

* **推理后处理（Postprocess）**
* **YOLO 输出解码**
* **NMS / 阈值过滤**
* **DeepStream 检测结果结构化转换**

### 为什么必须做


| 原因                | 说明                                                                |
| ------------------- | ------------------------------------------------------------------- |
| DeepStream 接口要求 | `nvinfer`的`custom-lib-path`只接受`.so`动态库                       |
| 功能必需            | YOLO 输出必须经过自定义解码，DeepStream 默认不会识别                |
| 性能关键            | 每帧调用一次，多路场景下必须在 C++ 中高效完成                       |
| 面试价值高          | 能直接体现你理解 YOLO 输出结构、DeepStream 推理接入方式和后处理流程 |

### 这一模块主要做什么

* 读取 YOLO 输出层
* 解码 bbox / class / confidence
* 做置信度筛选
* 做 NMS
* 输出 DeepStream 可消费的检测结果

### 说明

这一块不是“可选优化”，而是 **YOLO 接入 DeepStream 的标准工程做法**。
如果项目只选一个最核心的 C++ 模块优先落地，那一定是这个。

## 🔴 模块二：C++ Probe 热路径回调 / Meta Parser

### 是什么

当前系统中，Python 如果直接参与 `probe` 回调，对 `NvDsBatchMeta` 做高频遍历，会产生明显的性能开销，包括：

* `GstBuffer -> NvDsBatchMeta` 提取
* GLib linked list 遍历
* `pyds.cast()` 类型转换
* Python 小对象构造
* GIL 竞争

因此，将 **probe 热路径和元数据提取逻辑迁移到 C++**，是更合理的优化方式。

这一模块本质上属于：

* **推理结果提取**
* **元数据高效读取**
* **热路径降 Python 开销**
* **结果聚合前置层**

### 它主要做什么

* 从 `GstBuffer` 中直接读取 `NvDsBatchMeta`
* 遍历 frame meta / object meta
* 提取 `stream_id / frame_id / class_id / confidence / bbox / track_id / timestamp`
* `track_id` 按 `stream_id` 独立编号，OSD 中每一路视频都会从 `ID:1` 开始；原始 DeepStream 全局 ID 保留为 `global_track_id`
* 组织成轻量结果结构
* 交给 Python 非热路径模块消费

### 为什么值得做


| 原因                | 说明                                                   |
| ------------------- | ------------------------------------------------------ |
| Python 热路径开销大 | 多路场景下，Python probe 很容易吃掉 3–10ms 端到端延迟 |
| C++ 更贴近底层      | 可以直接访问 DeepStream C 结构体，避免跨语言反复装箱   |
| 收益明确            | 这是多路实时系统中很常见、也很有效的优化点             |
| 复杂度可控          | 相比跨进程共享内存，这一层更适合第一阶段先落地         |

### 设计建议

第一阶段不一定要直接做“共享内存 + lock-free ring buffer”。
更推荐先做：

* C++ 完成 `probe + meta parser`
* Python 只消费已经整理好的轻量结果

这样既保留：

* C++ 热路径优化
* Python 编排灵活性
* 系统可维护性

又避免第一阶段架构复杂度过高。

> “通过 profiling 发现 Python probe 回调是热路径瓶颈之一，因此将 `NvDsBatchMeta` 的遍历和目标结果提取迁移到 C++。这样 Python 不再直接参与高频元数据遍历，只负责消费已经整理好的结果对象，从而降低了 GIL 竞争和跨语言开销。”

## 🟡 模块三：轻量自定义 GStreamer / DeepStream 插件（结构下沉优化）

### 是什么

除了 parser 和 probe 之外，更进一步的工程化能力，是将某些局部处理逻辑直接做成 **轻量自定义 GStreamer / DeepStream 插件**，嵌入 pipeline 内部。

这个插件不追求“大而全”，而是追求：

* 单一职责
* 低耦合
* 直接嵌入流式处理链路
* 减少 Python 层参与

### 适合做什么

建议选择其中一个明确而轻量的场景：

* ROI 过滤插件
* 元数据筛选插件
* 轻量事件触发插件
* 多目标结果聚合插件
* 自定义 OSD 前的目标筛选插件

### 它主要做什么

* 作为 pipeline 内一个原生节点存在
* 直接处理 buffer / metadata
* 将局部逻辑从 Python 层下沉到 GStreamer/DeepStream 层

### 为什么这个思路好


| 原因             | 说明                                               |
| ---------------- | -------------------------------------------------- |
| 更像底层工程能力 | 不只是“调用 DeepStream”，而是“扩展 DeepStream” |
| 架构更自然       | 比“Python 调 C++ 再回 Python”更贴近流式系统设计  |
| 面试辨识度高     | 能体现你真正理解 GStreamer/DeepStream 插件模型     |

### 说明

这一模块比共享内存 ring buffer 更适合作为第三阶段的 C++ 亮点，因为：

* 技术价值高
* 工程表达更自然
* 不会过早引入过重的 IPC 复杂度

> “在完成 parser 和 probe 优化后，我进一步把局部业务逻辑下沉成轻量 GStreamer/DeepStream 插件，使这部分处理直接运行在 pipeline 内部，而不是依赖 Python 层调度。这能更好体现流式系统的原生扩展能力。”

## 🟡 模块四：Profiling 驱动的性能优化证据链（不是代码模块，但必须存在）

### 是什么

真正成熟的性能优化，不是“先写很多 C++”，而是先通过 profiling 找瓶颈，再决定优化点。

因此，本项目必须建立一套 **profiling 数据与性能证据链**，用来说明为什么要做 parser、probe 和插件优化。

### 主要做什么

记录和对比以下指标：

* 单路 / 多路 FPS
* 端到端 latency
* GPU 利用率
* CPU 利用率
* 丢帧率
* queue 堆积情况
* Python probe 与 C++ probe 的开销差异
* 优化前后吞吐对比

### 建议使用的手段

* `tegrastats`
* DeepStream latency measurement
* GStreamer tracing
* 自己埋点统计
* 推理前后对比实验

### 为什么这块必须有


| 原因       | 说明                                                   |
| ---------- | ------------------------------------------------------ |
| 决策依据   | 证明优化不是拍脑袋                                     |
| 性能闭环   | 说明改动前后到底提升了什么                             |
| 面试说服力 | 比“我做了优化”更有价值的是“我证明了为什么这样优化” |

> “我没有一开始就盲目下沉所有模块到 C++，而是先通过 profiling 识别瓶颈，再优先落地 parser 和 probe 热路径优化。这样每一个 C++ 模块都有明确的数据支撑和收益证明。”

## 🟡 预处理 / 后处理职责划分说明（关键设计原则）

### 推理前处理（Preprocess）

推理前处理包括：

* resize
* crop / ROI
* 颜色空间转换
* 数据布局变换
* 归一化

在本项目中建议按以下原则处理：

1. **优先使用 DeepStream / GStreamer 现成能力**
   * `nvstreammux`
   * `nvvidconv`
   * `nvdspreprocess`（如使用）
2. **只有在现成能力不足、或 profiling 明确表明存在瓶颈时**
   才考虑使用 C++/CUDA 自定义前处理模块

也就是说：

* 标准前处理：优先交给 DeepStream
* 强化前处理：必要时再用 CUDA / 自定义插件

### 推理后处理（Postprocess）

推理后处理主要包括：

* YOLO 输出解码
* bbox / class / confidence 恢复
* 阈值过滤
* NMS
* 转换成 DeepStream 目标结构

在本项目中，这部分的核心实现方式就是：

* **YOLO Custom Parser `.so`**

因此：

* **前处理不一定要自己写 C++**
* **后处理中的 parser 基本就是 C++ 必做项**

## 📊 四个优化模块的新优先级


| 模块                                     | 主要作用                            | 是否必做 | 第一阶段建议 |
| ---------------------------------------- | ----------------------------------- | -------- | ------------ |
| 模块一：YOLO Custom Parser`.so`          | 推理后处理 / YOLO 输出解析          | 是       | 必做         |
| 模块二：C++ Probe / Meta Parser          | 热路径元数据提取 / 降低 Python 开销 | 强烈推荐 | 优先做       |
| 模块三：轻量 GStreamer / DeepStream 插件 | 将局部逻辑下沉为原生 pipeline 节点  | 推荐     | 第二阶段做   |
| 模块四：Profiling 数据体系               | 说明为什么优化、优化效果如何        | 是       | 必做         |

## 🚫 当前阶段不建议优先重投入的方向

以下方向不是不能做，而是**不建议第一阶段优先上**：

* 跨进程共享内存 lock-free ring buffer
* 大规模自定义 CUDA kernel 重写全部前后处理
* 为了“显得底层”而重写已有 DeepStream 现成能力

这些更适合作为：

* 第二阶段高阶优化
* profiling 明确证明存在瓶颈之后再做
* 文档中的“后续演进方向”

## ✅ 推荐落地顺序

1. 完成 YOLO Custom Parser `.so`
2. 完成 C++ Probe / Meta Parser
3. 建立 profiling 指标采集与对比
4. 选择一个轻量自定义插件作为“底层扩展能力”展示点
5. 后续再视性能瓶颈决定是否引入共享内存 IPC 或 CUDA kernel 深化优化

## ✅ 一句话总结

> 本项目中的 C++/CUDA 优化，不是为了单纯“多写底层代码”，而是围绕 DeepStream 热路径与真实瓶颈做有优先级的工程化优化：第一阶段优先完成 YOLO parser 和 C++ probe 热路径优化，并用 profiling 数据证明收益；第二阶段再引入轻量插件和更深层的底层能力扩展。

## 十、优化建议总结


| 维度               | 现有状态        | 最新建议                                                                                                     | 优先级 |
| ------------------ | --------------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| **C++ 模块1**      | 架构已设计      | 实现 YOLO Custom Parser`.so`并编译部署                                                                       | 🔴 高  |
| **C++ 模块2**      | 架构已设计      | 实现 C++ Probe / Meta Parser，替代 Python 热路径遍历                                                         | 🔴 高  |
| **C++ 模块3**      | 原方案偏重      | 先不优先做共享内存 ring buffer，改为优先考虑轻量自定义 GStreamer / DeepStream 插件                           | 🟡 中  |
| **C++ 模块4**      | 原方案偏重 CUDA | 当前阶段不全面重写前处理/后处理，先让 DeepStream 组件承担标准前处理，仅在 profiling 证明瓶颈后再做 CUDA 深化 | 🟡 中  |
| **Profiling**      | 需要加强        | 建立优化前后 FPS / latency / GPU / CPU / 丢帧率对比体系                                                      | 🔴 高  |
| **前处理**         | 有设计          | 标准前处理优先交给 DeepStream / GStreamer，必要时再做 CUDA / 插件化增强                                      | 🟡 中  |
| **后处理**         | 有设计          | 后处理核心优先落到 YOLO parser`.so`                                                                          | 🔴 高  |
| **架构复杂度控制** | 有过度设计风险  | 第一阶段避免过早引入跨进程共享内存与过重 IPC 机制                                                            | 🔴 高  |

> **设计原则**：Python 管编排和配置，C++/CUDA 管热路径和底层。这是边缘AI部署岗最想看到的技术组合。

## 🟡 模块四：CUDA 预处理/后处理

### 是什么

在 GPU 上直接写 CUDA kernel 处理图像预处理和后计算，减少 CPU-GPU 数据搬运和 `nvstreammux`/`nvdsosd` 的功能限制。

### 适用场景


| Kernel                   | 用途                    | 替代什么                    |
| ------------------------ | ----------------------- | --------------------------- |
| `roi_crop_resize_kernel` | 自定义 ROI 裁剪 + 缩放  | `nvstreammux` 的固定 resize |
| `nms_kernel`             | 置信度过滤 + NMS 后处理 | Python/C++ CPU 端 NMS       |
| `osd_overlay_kernel`     | 自定义画框/OSD 叠加     | `nvdsosd` 固定样式          |
| `rgb_to_nv12_kernel`     | 色彩空间转换            | GStreamer videoconvert      |

### 核型实现: ROI 裁剪 + Bilinear Resize

```cuda
// custom_libs/cuda_kernels/preprocess.cu

#include <cuda_runtime.h>

// ROI 区域描述符
struct RoiRegion {
    float left, top, width, height;   // 归一化坐标 [0, 1]
};

// YUV420/NV12 平面 ROI 裁剪 + bilinear resize
// 输入: NV12 格式 (Y plane + interleaved UV plane)
// 输出: 指定尺寸的 planar RGB (可直接输入 TensorRT)
__global__ void roi_crop_and_resize_kernel(
    const uint8_t* __restrict__ y_plane,      // Y 平面
    int y_pitch,                               // Y 平面行步长 (bytes)
    const uint8_t* __restrict__ uv_plane,      // interleaved UV 平面
    int uv_pitch,
    int src_width, int src_height,
    // --- ROI 参数 ---
    const RoiRegion* __restrict__ rois,
    int num_rois,
    // --- 输出 ---
    float* __restrict__ output,                // planar RGB, 归一化到 [0, 1]
    int out_width, int out_height)
{
    // 每个 block 处理一个 ROI，thread 并行处理像素
    int roi_idx  = blockIdx.x;
    int out_x    = blockIdx.y * blockDim.x + threadIdx.x;
    int out_y    = blockIdx.z * blockDim.y + threadIdx.y;

    if (roi_idx >= num_rois) return;
    if (out_x >= out_width || out_y >= out_height) return;

    const RoiRegion& roi = rois[roi_idx];

    // 计算源图像中的浮点坐标 (bilinear interpolation)
    float src_x = (roi.left + (float)out_x / out_width  * roi.width)  * src_width;
    float src_y = (roi.top  + (float)out_y / out_height * roi.height) * src_height;

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = min(x0 + 1, src_width  - 1);
    int y1 = min(y0 + 1, src_height - 1);
    float wx = src_x - x0;
    float wy = src_y - y0;

    // --- Y 通道 bilinear ---
    float y00 = (float)y_plane[y0 * y_pitch + x0];
    float y10 = (float)y_plane[y0 * y_pitch + x1];
    float y01 = (float)y_plane[y1 * y_pitch + x0];
    float y11 = (float)y_plane[y1 * y_pitch + x1];
    float Y = (1-wy)*((1-wx)*y00 + wx*y10) + wy*((1-wx)*y01 + wx*y11);

    // --- UV 通道 bilinear ---
    int uv_x0 = x0 / 2, uv_x1 = min(uv_x0 + 1, src_width  / 2 - 1);
    int uv_y0 = y0 / 2, uv_y1 = min(uv_y0 + 1, src_height / 2 - 1);
    float uv_wx = (src_x / 2.0f) - uv_x0;
    float uv_wy = (src_y / 2.0f) - uv_y0;

    float u00 = (float)uv_plane[uv_y0 * uv_pitch + uv_x0 * 2];
    float v00 = (float)uv_plane[uv_y0 * uv_pitch + uv_x0 * 2 + 1];
    float u10 = (float)uv_plane[uv_y0 * uv_pitch + uv_x1 * 2];
    float v10 = (float)uv_plane[uv_y0 * uv_pitch + uv_x1 * 2 + 1];
    float u01 = (float)uv_plane[uv_y1 * uv_pitch + uv_x0 * 2];
    float v01 = (float)uv_plane[uv_y1 * uv_pitch + uv_x0 * 2 + 1];
    float u11 = (float)uv_plane[uv_y1 * uv_pitch + uv_x1 * 2];
    float v11 = (float)uv_plane[uv_y1 * uv_pitch + uv_x1 * 2 + 1];

    float U = (1-uv_wy)*((1-uv_wx)*u00 + uv_wx*u10) + uv_wy*((1-uv_wx)*u01 + uv_wx*u11);
    float V = (1-uv_wy)*((1-uv_wx)*v00 + uv_wx*v10) + uv_wy*((1-uv_wx)*v01 + uv_wx*v11);

    // --- YUV → RGB ---
    float R = Y + 1.402f   * (V - 128.0f);
    float G = Y - 0.34414f * (U - 128.0f) - 0.71414f * (V - 128.0f);
    float B = Y + 1.772f   * (U - 128.0f);

    int out_offset = (roi_idx * 3 * out_height + 0) * out_width + out_y * out_width + out_x;
    output[out_offset + 0 * out_height * out_width] = R / 255.0f;
    output[out_offset + 1 * out_height * out_width] = G / 255.0f;
    output[out_offset + 2 * out_height * out_width] = B / 255.0f;
}

// Host 封装
extern "C" cudaError_t launch_roi_crop_kernel(
    const uint8_t* y_plane, int y_pitch,
    const uint8_t* uv_plane, int uv_pitch,
    int src_width, int src_height,
    const RoiRegion* rois, int num_rois,
    float* output, int out_width, int out_height,
    cudaStream_t stream)
{
    dim3 block(32, 4);
    dim3 grid(num_rois,
              (out_width  + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y);

    roi_crop_and_resize_kernel<<<grid, block, 0, stream>>>(
        y_plane, y_pitch, uv_plane, uv_pitch,
        src_width, src_height,
        rois, num_rois,
        output, out_width, out_height);

    return cudaGetLastError();
}
```

### 核型实现: 自定义 OSD 叠加

```cuda
// custom_libs/cuda_kernels/osd_overlay.cu

// 每个线程画一个检测框的一条水平线
__global__ void draw_bbox_overlay_kernel(
    uint8_t* __restrict__ y_plane,
    int y_pitch,
    int img_width, int img_height,
    const BBox* __restrict__ bboxes,
    const int* __restrict__ class_ids,
    const uint64_t* __restrict__ track_ids,
    int num_detections,
    float draw_thickness)
{
    int det_idx = blockIdx.x;
    if (det_idx >= num_detections) return;

    const BBox& box = bboxes[det_idx];
    int x1 = max(0, (int)box.left);
    int y1 = max(0, (int)box.top);
    int x2 = min(img_width  - 1, (int)(box.left + box.width));
    int y2 = min(img_height - 1, (int)(box.top  + box.height));

    int line_idx = threadIdx.x;
    int thickness = max(2, (int)draw_thickness);

    // 四条边：0=top, 1=bottom, 2=left, 3=right
    int total_lines = (y2 - y1) * 2 + (x2 - x1) * 2;
    if (line_idx >= total_lines) return;

    // 定位在四条边上的像素，写入高亮值 (255 = 白)
    // ...（具体坐标计算省略，原理同上一个 kernel）
}

extern "C" cudaError_t launch_osd_kernel(
    uint8_t* nvmm_y_plane, int y_pitch,
    int width, int height,
    const BBox* d_bboxes, const int* d_class_ids,
    const uint64_t* d_track_ids, int num_detections,
    cudaStream_t stream);
```

## 📊 四个 C++ 模块总览

```
DeepStream Pipeline (推理进程 — 绑定 GPU)
│
├── [解码] NVDEC (硬件)
│
├── [批处理] nvstreammux (硬件)
│
├── [推理] nvinfer + TensorRT
│    └── 🔴 模块一: YOLO Custom Parser .so  ← 推理输出解析, .so 由 nvinfer 加载
│
├── [跟踪] nvtracker (硬件)
│
├── [Probe] 元数据提取
│    └── 🔴 模块二: C++ Probe 热路径回调  ← 每帧 meta 遍历, 零 Python 开销
│         │
│         │  DetectionResult (POD, 64B)
│         ▼
│    🟡 模块三: C++ Lock-Free Shm Ring Buffer  ← 共享内存结果总线
│         │
│   ══════╪════════ 进程边界 (IPC) ═════════╪══════
│         │
│         ▼  消费者进程们 (同一台 Jetson 上)
│    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│    │ Web Dashboard│  │ JSON Writer   │  │ MQTT/Kafka    │
│    │ (FastAPI)   │  │ (文件落盘)    │  │ Publisher     │
│    └─────────────┘  └──────────────┘  └──────────────┘
│
├── [预处理] 可选自定义预处理
│    └── 🟡 模块四: CUDA ROI/Kernel  ← GPU 原生处理
│
├── [OSD] 可选自定义叠加
│    └── 🟡 模块四: CUDA OSD Kernel  ← GPU 原生画框
│
├── [编码] NvV4L2H264Enc (硬件)
│
└── [输出] rtmpsink / flvmux (标准 GStreamer, 不变)
```

# 八、预期性能指标


| 指标      | 目标             |
| --------- | ---------------- |
| 并发路数  | 6–12路 2K       |
| 延迟      | 80–200ms        |
| FPS       | 接近输入70–90%  |
| GPU利用率 | 60–90%稳定      |
| 丢帧率    | <5%              |
| 性能提升  | 2–4×（优化后） |

# 九、项目价值总结

## ✔️一句话版本

> 该项目构建了基于 DeepStream 的多路视频边缘推理系统，通过 TensorRT 加速与工程级优化，实现多路视频实时目标检测与跟踪，并显著提升推理吞吐与系统稳定性。

## ✔️核心能力证明点

这个项目可以证明你具备：

* 边缘AI系统设计能力
* GPU推理优化能力
* DeepStream工程能力
* 高并发视频处理能力
* 系统级性能调优能力


# 十、优化建议总结


| 维度               | 现有状态        | 最新建议                                                                                                     | 优先级 |
| ------------------ | --------------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| **C++ 模块1**      | 架构已设计      | 实现 YOLO Custom Parser`.so`并编译部署                                                                       | 🔴 高  |
| **C++ 模块2**      | 架构已设计      | 实现 C++ Probe / Meta Parser，替代 Python 热路径遍历                                                         | 🔴 高  |
| **C++ 模块3**      | 原方案偏重      | 先不优先做共享内存 ring buffer，改为优先考虑轻量自定义 GStreamer / DeepStream 插件                           | 🟡 中  |
| **C++ 模块4**      | 原方案偏重 CUDA | 当前阶段不全面重写前处理/后处理，先让 DeepStream 组件承担标准前处理，仅在 profiling 证明瓶颈后再做 CUDA 深化 | 🟡 中  |
| **Profiling**      | 需要加强        | 建立优化前后 FPS / latency / GPU / CPU / 丢帧率对比体系                                                      | 🔴 高  |
| **前处理**         | 有设计          | 标准前处理优先交给 DeepStream / GStreamer，必要时再做 CUDA / 插件化增强                                      | 🟡 中  |
| **后处理**         | 有设计          | 后处理核心优先落到 YOLO parser`.so`                                                                          | 🔴 高  |
| **架构复杂度控制** | 有过度设计风险  | 第一阶段避免过早引入跨进程共享内存与过重 IPC 机制                                                            | 🔴 高  |

> **设计原则**：Python 管编排和配置，C++/CUDA 管热路径和底层。这是边缘AI部署岗最想看到的技术组合。
>

### 🎯 核心任务一：直观展示系统核心能力

这是前端最基础、最重要的任务。它的目标是让观看者（面试官、业务方）一眼看懂你做了什么。

1. **多路视频流实时播放**：页面最核心的区域是视频显示区，**并排显示多路视频流**。这直接证明了你的系统能成功接入并处理6-12路视频。
2. **叠加可视化信息**：在视频帧上**叠加绘制检测框、跟踪ID、类别标签和置信度**。这验证了你的检测和跟踪模块工作正常，且结果在工程上是准确的。
3. **显示关键性能指标**：在页面侧边栏或顶部，实时刷新系统运行的核心KPI，如**当前接入路数、整体FPS、平均推理延迟、GPU利用率和丢帧率**。这能证明你的系统运行在预期的性能指标上，且状态健康。

### 🔗 核心任务二：实现数据闭环与结构化展示

这个任务是为了证明你具备全链路开发能力和数据价值挖掘意识。

1. **结构化事件列表**：在页面下方或另一侧，设计一个**滚动的事件/告警日志**。每当系统检测到目标（如“人员”或“车辆”），就生成一条结构化记录，包含**时间、摄像头ID、目标类型、跟踪ID、所在区域**等信息。这展示了你的系统能产生有价值的业务数据。
2. **简单交互能力**：可以通过按钮或下拉菜单，筛选查看特定摄像头的事件记录。这证明你的前后端数据是贯通的，具备初步的可交互性。

### 🎚️ 核心任务三：提供基础的状态控制与反馈

这个任务是为了让你展示系统工程思维和对边缘设备的操作理解。

1. **显示系统状态**：清晰标明当前显示的视频源是“离线文件”还是“实时RTSP流”，如果使用RTSP，可以显示连接状态（在线/离线），这体现了对工业视频流管理的理解。
2. **基础控制选项**：实现一个简单的开关按钮，如 **“开启/停止检测”** 或 **“切换显示/原始画面”**。这是展示你对后端Pipeline有控制权，而不是只能跑一个固定脚本。
