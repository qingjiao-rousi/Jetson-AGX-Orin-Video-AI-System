# Jetson 端侧部署与边缘 AI 深化学习路线

## 1. 研究方向

本项目后续深入方向确定为：

> 面向 Jetson AGX Orin 的多路视觉模型端侧部署、量化和性能优化。

重点不是单独训练一个更大的模型，而是研究如何把视觉模型可靠、高效地部署到资源受限的边缘设备上：

```text
模型 -> ONNX -> TensorRT engine -> DeepStream 多路调度
-> GPU/显存/功耗优化 -> 稳定的端侧生产服务
```

## 2. 学习目标

完成后应能够独立回答并验证以下问题：

- 一个 PyTorch/YOLO 模型如何导出为可部署的 ONNX；
- ONNX 如何构建为针对 Jetson 硬件优化的 TensorRT engine；
- FP32、FP16、INT8 和混合精度的精度、速度、显存、功耗差异；
- 动态 batch、TensorRT profile 和 DeepStream `nvstreammux` 如何配合；
- 多路解码、推理、tracker、OSD 和输出编码的真正瓶颈在哪里；
- 如何在 GPU、温度、功耗和延迟约束下选择部署策略；
- 如何让模型部署具备版本管理、自动恢复、日志和回滚能力。

## 3. 分阶段路线

### 阶段一：建立 FP16 部署基线

目标是理解当前模型从 ONNX 到 TensorRT engine 的完整路径。

学习内容：

- YOLO ONNX 输入输出结构；
- TensorRT engine、optimization profile 和动态 batch；
- `trtexec` 的构建与性能测试；
- Jetson AGX Orin 的 GPU、显存和功耗监控。

项目实验：

```bash
trtexec --loadEngine=models/yolov8s.engine \
  --shapes=input:8x3x640x640 \
  --duration=10 \
  --dumpOptimizationProfile
```

需要记录：

- engine 构建参数和 TensorRT 版本；
- batch 1、4、8 的 TensorRT 单模型吞吐；
- GPU 利用率、显存、温度和功耗；
- DeepStream 端到端 FPS 与 `trtexec` 理论吞吐的差距。

### 阶段二：INT8 量化

目标是理解量化带来的速度和精度变化。

学习内容：

- PTQ 与 QAT 的区别；
- calibration dataset 的选择；
- 激活值范围、量化 scale 和校准缓存；
- INT8 engine 的精度验证；
- 量化导致的漏检、误检和 confidence 变化。

项目实验：

1. 固化当前 FP16 结果作为基线；
2. 准备覆盖不同光照、距离、目标大小和场景的校准集；
3. 构建 INT8 engine；
4. 在相同输入和相同阈值下比较 FP16/INT8。

必须同时记录：

```text
mAP / precision / recall
误检率 / 漏检率
总 FPS / 单路 FPS
GPU / 显存 / 功耗 / 温度
```

### 阶段三：混合精度逐层优化

目标是找到精度和性能之间更好的平衡，而不是简单地把所有层都改成 INT8。

研究方法：

- 先构建全 INT8 engine；
- 分析每层的精度敏感性和耗时；
- 将敏感层保留为 FP16；
- 将稳定且耗时高的层改为 INT8；
- 比较不同 precision profile 的收益。

最终形成类似以下实验表：

| Profile | 精度 | 总 FPS | 单路 FPS | GPU | 功耗 | 温度 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 baseline |  |  |  |  |  |  |  |
| INT8 all layers |  |  |  |  |  |  |  |
| mixed precision A |  |  |  |  |  |  |  |
| mixed precision B |  |  |  |  |  |  |  |

### 阶段四：DeepStream 多路性能优化

目标是理解模型之外的端到端瓶颈。

重点研究：

- `nvstreammux` batch size 和 batched push timeout；
- 多路 NVDEC 解码吞吐；
- queue、leaky queue 和 backpressure；
- tracker 对 GPU/CPU 的影响；
- OSD 和视频编码开销；
- Python probe 与 C++ parser 的性能差异；
- zero-copy 和 NVMM 内存路径。

实验方式：一次只改变一个因素，并比较 1/4/8 路：

```text
推理-only fake sink
推理 + tracker
推理 + tracker + OSD
推理 + tracker + OSD + H.264 编码
```

### 阶段五：端侧生产化

目标是让部署结果可以长期运行和维护。

学习内容：

- systemd 服务和异常自动重启；
- engine、模型和配置版本绑定；
- 日志轮转与输出保留策略；
- 温度、功耗和内存异常处理；
- 模型/engine 回滚；
- 远程升级和部署检查。

## 4. 与当前项目的对应关系

| 学习主题 | 项目对应内容 |
| --- | --- |
| TensorRT engine | `models/*.onnx`、`models/*.engine`、`scripts/build_yolov8s_engine_batch8.sh` |
| 多路调度 | `PipelineBuilder`、`nvstreammux`、`SourceFactory` |
| 结果解析 | `MetaParser`、YOLO custom parser、`ProbeRegistry` |
| 异步输出 | `JsonWriter`、runtime metrics |
| 性能监控 | `tegrastats`、`RuntimeMetricsRecorder` |
| 端侧服务 | `deploy/`、systemd 安装脚本、输出清理脚本 |
| 质量验收 | `scripts/check_rtsp_inproc_outputs.py`、性能和稳定性报告 |
| 多模型扩展 | `Orchestrator`、模型配置、统一帧结果和 fusion 层 |

## 5. 最终能力目标

完成这条路线后，应能够交付一个可解释、可复现的端侧部署结果：

```text
同一个模型
-> 多种 precision profile
-> 多种 batch profile
-> 多种输入路数
-> 可比较的精度、FPS、功耗、温度和稳定性数据
```

最终成果不只是一个 `.engine` 文件，而是：

- 可重复的模型导出和 engine 构建流程；
- FP16、INT8、混合精度实验报告；
- 1/4/8 路端到端性能报告；
- 精度与性能的权衡结论；
- 可长期运行的 Jetson 部署服务；
- 支持后续多模型扩展的统一推理架构。

## 6. 研究原则

1. 先建立基线，再做优化。
2. 一次只改变一个变量。
3. 精度下降必须用数据证明，不能只看 FPS。
4. TensorRT 单模型性能不能替代 DeepStream 端到端性能。
5. GPU 利用率高不等于系统吞吐最优，还要观察丢帧、延迟、温度和功耗。
6. 所有 engine 必须记录构建设备、JetPack、CUDA、TensorRT、模型和配置版本。
