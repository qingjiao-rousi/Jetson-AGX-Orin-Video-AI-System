# AI 学习知识文档生成需求说明

## 1. 使用目的

请 AI 围绕以下方向，为我生成一套系统、连续、可实践的学习知识文档：

> Jetson AGX Orin 上的端侧 AI、视觉模型部署、TensorRT、DeepStream、多路推理、模型量化和边缘设备性能优化。

我的目标不是只了解概念，而是能够：

- 看懂端侧模型部署的完整链路；
- 独立完成 ONNX、TensorRT engine 和 DeepStream 部署；
- 理解 FP32、FP16、INT8 和混合精度的差异；
- 分析 GPU、显存、功耗、温度、延迟和 FPS；
- 找到多路视频推理的真实瓶颈；
- 设计稳定的端侧生产部署系统；
- 在现有项目中继续加入新模型和多模型推理。

## 2. 我的项目背景

项目基于：

- Jetson AGX Orin 32GB；
- Ubuntu + JetPack + CUDA + TensorRT；
- DeepStream + GStreamer；
- YOLOv8 TensorRT engine；
- 多路本地 MP4 和 RTSP 输入；
- MP4 模拟 RTSP 摄像头；
- `nvstreammux` 多路合批；
- tracker、OSD、H.264 编码；
- MP4、RTMP、MediaMTX RTSP relay 输出；
- JSONL 和 runtime metrics；
- 后续需要加入新的模型进行多模型推理。

当前项目应被视为一个“端侧多路视觉推理系统”，而不是单纯的目标检测 demo。

## 3. 知识文档必须覆盖的主题

### 3.1 端侧 AI 基础

- 云端推理、边缘推理和端侧推理的区别；
- Jetson 的 GPU、CPU、内存、显存、NVDEC、NVENC、DLA；
- 延迟、吞吐、FPS、实时性、功耗和热设计；
- 为什么端侧优化需要精度、性能和稳定性联合考虑。

### 3.2 模型部署完整链路

```text
PyTorch/YOLO
-> ONNX
-> TensorRT parser
-> TensorRT engine
-> DeepStream nvinfer
-> tracker/OSD
-> 多路输入输出
```

需要解释：

- 每一步解决什么问题；
- 输入输出 tensor 如何变化；
- 静态 batch 和动态 batch 的区别；
- engine 为什么与 GPU、JetPack、TensorRT 版本相关；
- 常见构建失败和运行时 warning 如何排查。

### 3.3 TensorRT

- network、layer、tensor、binding、profile、workspace、tactic；
- FP32、FP16、INT8、BF16 的区别；
- `trtexec` 常用参数；
- dynamic shape 和 optimization profile；
- batch 1、4、8 的性能测试；
- engine 构建、加载、校验、备份和回滚；
- TensorRT 单模型 benchmark 与 DeepStream 端到端 benchmark 的区别。

### 3.4 INT8 与混合精度量化

- PTQ 和 QAT；
- calibration dataset 的作用；
- 激活值范围、scale、zero point；
- 全 INT8 为什么可能造成漏检和误检；
- 哪些层通常对量化敏感；
- 如何让敏感层保留 FP16；
- 如何设计逐层精度实验；
- 如何同时评价精度、FPS、功耗、温度和显存。

必须给出可执行的实验设计，而不是只解释概念。

### 3.5 DeepStream 和 GStreamer

- source、decoder、NVMM、`nvstreammux`、`nvinfer`、tracker、OSD、encoder、sink；
- dynamic pad 和多路 branch；
- batch timeout、live-source、PTS、NTP timestamp；
- queue、leaky queue、backpressure 和丢帧；
- zero-copy 和 CPU/GPU 内存拷贝；
- Python probe、C++ parser 和 CUDA 热路径；
- MP4、RTSP、RTMP 输出链路的差异。

### 3.6 性能分析和 profiling

必须教我如何使用并解释：

- `tegrastats`；
- `trtexec`；
- GStreamer latency tracing；
- Nsight Systems；
- Nsight Compute；
- CPU、GPU、显存、内存、温度和功耗指标。

每一个指标都要说明：

```text
它是什么 -> 如何采集 -> 如何判断异常 -> 如何定位原因 -> 如何优化
```

### 3.7 多路、多模型和调度

- 多路输入为什么需要 batch；
- batch size 和 batch timeout 如何权衡；
- 多模型共享解码和主循环的方式；
- 每个模型如何独立配置和统计；
- `model_id + stream_id + frame_id + timestamp` 的结果关联；
- 多模型串行、并行、交替调度的区别；
- 多模型结果 fusion 层如何设计。

### 3.8 端侧生产化

- systemd；
- 自动重启和断流恢复；
- 日志轮转；
- 输出保留和清理；
- engine、模型、配置版本管理；
- 设备温度和功耗保护；
- 远程升级、回滚和故障排查。

## 4. 每篇知识文档的固定结构

AI 每次输出一个主题时，必须按照以下结构：

1. 本篇学习目标；
2. 为什么端侧部署需要这个知识；
3. 核心概念；
4. 从输入到输出的完整流程；
5. 与我当前项目的对应关系；
6. 最小可运行示例；
7. Jetson 上的实际命令；
8. 指标和日志如何观察；
9. 常见错误和排查步骤；
10. 一个实验任务；
11. 实验记录表；
12. 本篇小结；
13. 复习问题；
14. 下一篇学习建议。

## 5. 对 AI 输出质量的要求

- 使用中文，保留关键英文术语；
- 先解释整体，再深入局部；
- 不要只给定义，要解释为什么；
- 不要假设我已经掌握 TensorRT 或 DeepStream；
- 代码和命令必须说明执行环境；
- 区分“理论上可行”和“在当前项目中已验证”；
- 不要编造项目中不存在的文件、配置或测试结果；
- 涉及版本、参数或命令时，明确可能存在的 JetPack/TensorRT 差异；
- 每个重要结论都给出验证方法；
- 每次只深入一个主题，不要一次堆砌所有内容；
- 发现我的理解错误时，直接指出并解释原因。

## 6. 推荐生成顺序

```text
1. 端侧 AI 总览
2. Jetson 硬件和资源模型
3. YOLO 到 ONNX
4. ONNX 到 TensorRT engine
5. FP32/FP16/INT8/混合精度
6. trtexec 和 TensorRT profiling
7. DeepStream pipeline
8. nvstreammux 多路合批
9. tracker/OSD/编码和输出
10. 端到端性能分析
11. 多模型调度与 fusion
12. systemd 和端侧生产化
```

## 7. 最终学习成果

需要最终形成：

- 一套端侧 AI 基础知识文档；
- 一套 TensorRT/DeepStream 部署文档；
- 一套 FP16/INT8/混合精度实验报告；
- 一套 1/4/8 路性能对比报告；
- 一套多模型推理架构设计文档；
- 一套 Jetson 生产部署和故障排查手册。
