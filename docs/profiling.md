# TensorRT 与 Nsight Profiling

本项目将 profiling 分成两层。TensorRT profile 用于测量独立主模型 engine 的层级耗时；
Nsight Systems 用于测量真实 DeepStream pipeline 中的 CPU 线程、CUDA kernel、内存复制和
Python worker 的时间线。两者回答的问题不同，不能互相替代。

所有报告、CSV、`.nsys-rep` 文件和运行输出均是本机生成物，不能提交 Git。

## 前提

- 在 Jetson AGX Orin 上运行，engine 必须由当前 TensorRT 构建。
- 用 `nvpmodel -q` 记录功耗模式；profiling 前后保持散热与时钟状态一致。
- Nsight Systems CLI 命令 `nsys` 必须在 `PATH`。若脚本提示找不到，安装与 JetPack 版本
  匹配的 Nsight Systems CLI 后再执行，不能用其他版本报告替代。

## TensorRT 层级 Profile

```bash
cd /home/nvidia/Desktop/YOLO
scripts/benchmark/profile_primary_tensorrt.sh \
  --batch-size 8 \
  --duration 30 \
  --output-dir outputs/profiling/tensorrt_primary_b8
```

输出包括 FP16/INT8 各自的 `profile.json`、`times.json` 和终端日志。比较 layer profile 时，
关注耗时最高的层、`Reformat`/`Transpose`/类型转换、检测头与总 GPU compute time。它不含
视频解码、`nvstreammux`、tracker、Python worker、JSON writer 或输出 sink，因此不能将其
吞吐直接写为系统 FPS。

必要时可再运行 batch 1，确认主模型 batch profile 对每帧耗时的影响：

```bash
scripts/benchmark/profile_primary_tensorrt.sh --batch-size 1 --duration 30
```

## Nsight Systems 时间线

使用当前已验证的独立队列配置、PPE batch 1，采集 60 秒 `fake` sink：

```bash
scripts/benchmark/profile_pipeline_nsys.sh \
  --config configs/app/app_multifile_8_primary_int8_isolated_tasks.yaml \
  --run-seconds 60 \
  --sink fake \
  --output-dir outputs/profiling/nsys_primary_int8
```

该脚本输出 `.nsys-rep` 和 `cuda_gpu_kern_sum`、`cuda_api_sum`、`osrt_sum` CSV。用 Nsight
Systems GUI 打开 `.nsys-rep`，按以下顺序检查：

1. GPU 上主模型 CUDA kernel 是否长期空闲，或被 PPE/Pose/Fire worker kernel 穿插。
2. CPU 上 GStreamer streaming thread、Python worker、JSON writer 是否出现长时间等待或
   单核饱和。
3. CUDA API 中 `cudaMemcpy*`、`cudaStreamSynchronize` 是否占比异常，判断内存复制/同步
   是否限制 worker。
4. CUDA kernel 时间线是否存在主模型与专用模型无法重叠的长串行区段。

Nsight 本身有采样和 trace 开销，因此该运行只用于定位，不能与 benchmark 表中的 FPS/P95
直接比较。性能结论必须回到未插桩的 benchmark 配置中复测。

## 结果记录模板

| 项目 | 待填结果 | 结论 |
| --- | --- | --- |
| 主模型 FP16/INT8 layer profile | `[待实测]` | `[待判断]` |
| 主模型 batch 1/8 GPU compute time | `[待实测]` | `[待判断]` |
| GPU 空闲/串行区段 | `[待实测]` | `[待判断]` |
| CPU 热线程或同步 API | `[待实测]` | `[待判断]` |
| 下一项优化是否值得做 | `[待实测]` | `[待判断]` |

没有 profile 证据前，不应宣称具体 kernel、Python GIL、内存带宽或 CUDA 同步是主要瓶颈。
