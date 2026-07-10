# AGX 最终目标 vs 当前状态 — 差距分析

> **活文档说明**：本文档随项目讨论实时更新。每次讨论涉及进度变化、新发现的问题或决策调整后，同步更新对应章节。
>
> **基准文档**：[AGX.md](AGX.md) — 定义了项目的最终架构目标
>
> **最后更新**：2026-07-09

## 一、架构模式 — 最大差距


| 维度          | AGX.md 最终目标                                       | 当前状态 (2026-07-09)                           |
| ------------- | ----------------------------------------------------- | ----------------------------------------------- |
| Pipeline 模式 | **单 pipeline 内 nvstreammux 合批** (batch-size=6-12) | **8 个独立 deepstream 进程**，每进程只处理 1 路 |
| 输入源        | 6-12 路**RTSP 实时流**                                | 本地 MP4 文件                                   |
| 模型加载      | 1 份 TensorRT engine 共享                             | 8 个进程各自加载 1 份 engine，GPU 显存 ×8      |

这是最根本的架构差距。当前 `BATCH_JOBS=8` 是"开 8 个独立进程"，不是单 pipeline 的 `nvstreammux batch-size=8`。

## 二、功能模块对照


| AGX.md 模块  | 目标                   | 当前状态                  | 差距                   |
| ------------ | ---------------------- | ------------------------- | ---------------------- |
| 多路视频接入 | RTSP 实时流            | 本地 MP4                  | ❌ RTSP 未接入         |
| 硬件解码     | NVDEC                  | NVDEC ✅                  | —                     |
| 多路批处理   | nvstreammux batch=6-12 | 无合批 (独立进程)         | ❌                     |
| YOLO 推理    | YOLOv8n FP16 → INT8   | YOLOv8s FP16              | 🟡 模型不同, INT8 未做 |
| 目标跟踪     | IOU → NvDCF           | IOU                       | 🟡 NvDCF 未做          |
| OSD 叠加     | nvdsosd                | nvdsosd ✅                | —                     |
| 编码输出     | NvV4L2H264Enc → RTMP  | NvV4L2H264Enc → 本地 MP4 | 🟡 RTMP 推流未做       |
| 结构化输出   | JSON/MQTT/Kafka        | JSONL                     | 🟡 MQTT/Kafka 未做     |

## 三、C++ 加速模块对照（AGX.md 核心层）


| C++ 模块                           | 目标     | 当前状态          | 差距                      |
| ---------------------------------- | -------- | ----------------- | ------------------------- |
| **模块一: YOLO Custom Parser .so** | 必做     | ✅ 已有预编译 .so | 有二进制，无源码          |
| **模块二: C++ Probe/Meta Parser**  | 强烈推荐 | ❌ 未实现         | Python probe 仍在热路径上 |
| **模块三: 轻量 GStreamer 插件**    | 第二阶段 | ❌ 未开始         | AGX.md 中有设计           |
| **模块四: CUDA Kernel**            | 第二阶段 | ❌ 未开始         | AGX.md 中有伪代码         |

## 四、优化控制模块对照


| 优化能力         | 目标                           | 当前状态                                     | 差距                  |
| ---------------- | ------------------------------ | -------------------------------------------- | --------------------- |
| GPU 负载监控     | tegrastats 真实数据            | `gpu_monitor.py` 返回占位字典                | ❌ 无真实数据         |
| 丢帧策略         | **有界缓冲队列**（buffer满→丢） | `fps_controller` 概率丢帧 + `backpressure_controller` 消费者缺失 | 🔴 方案需切换（见下方 4.1） |
| Profiling 证据链 | FPS/latency/GPU/CPU/丢帧率对比 | 无系统化 profiling                           | ❌ 未建立             |

### 4.1 丢帧策略设计决策：有界缓冲队列替代自适应概率丢帧

**决策**：自适应概率丢帧（`fps_controller`）依赖 GPU 利用率和队列深度做预测，但 GPU 数据源是占位值，`backpressure_controller` 消费者端也未闭环。决定改为**有界缓冲队列（bounded buffer）**方案：在 probe 回调（生产者）和 JSON writer（消费者）之间插入固定容量的 `queue.Queue`，队列满时直接丢弃新帧。

**原理对比**：

```
当前方案（预测式）:
  probe → parse → fps_controller(假GPU数据) → 概率丢帧 → 同步write
  
新方案（反应式）:
  probe → parse → queue.put_nowait() ──→ 消费者线程 → write
                    ↑ 队列满抛 Full
                    丢帧（确定性, 无需 GPU 指标）
```

**为什么更好**：

| 维度 | 自适应概率丢帧（旧） | 有界缓冲队列（新） |
|------|---------------------|-------------------|
| 依赖 | GPU 指标（当前是假的） | 无外部依赖 |
| 触发条件 | `random() < drop_rate`（概率） | `queue.Full`（确定性） |
| 突发容忍 | 无缓冲，瞬时峰值直接丢 | buffer 吸收短期突发 |
| 可观测性 | drop_rate/avg_gpu（间接） | `queue.qsize()`（直接反映系统压力） |
| 复杂度 | 阶梯式调整 ±0.15/±0.05，需调参 | `maxsize=N` 一个参数 |

**实现要点**：

```python
# orchestrator.py — 热路径
def on_frame_result(self, result: FrameResult) -> None:
    parsed = self.meta_parser.parse(result)
    try:
        self._result_queue.put_nowait(parsed)
    except queue.Full:
        self._dropped_count += 1
        if self._dropped_count % 100 == 0:
            logging.warning("queue full, dropped %s frames", self._dropped_count)

# bootstrap.py — 启动消费者线程
self._result_queue = queue.Queue(maxsize=settings.optimization.max_queue_size)
self._consumer_thread = threading.Thread(
    target=self._consume_results, daemon=True
)
self._consumer_thread.start()

def _consume_results(self):
    while not self._stop_event.is_set():
        try:
            parsed = self._result_queue.get(timeout=0.2)
            self.json_writer.write(parsed)
            self._result_queue.task_done()
        except queue.Empty:
            continue
```

**被替代的模块**：此方案落地后，`fps_controller.py` 和 `backpressure_controller.py` 的热路径角色被 `queue.Queue` 取代。它们可降级为**监控辅助模块**（统计丢帧数、记录队列深度快照、提供 `stats()` 接口），但不再做丢帧决策。

**与 demo 项目的对应**：本质上是 Rockchip 项目中 `Mbuffer`（`images[i]` + `mutexes[i]` 固定缓冲区 + `std::move` + `buffer.img.empty()` 检查）的 Python 等效实现。`StreamLoader::operator()()` 是生产者，`combineImage()` 是消费者，中间的共享 `images` 数组就是有界缓冲。


## 五、Web 能力对照


| Web 能力     | 目标                    | 当前状态  | 差距                             |
| ------------ | ----------------------- | --------- | -------------------------------- |
| 离线验收看板 | —                      | ✅ 已完成 | 批量结果浏览、质量状态、视频播放 |
| 实时视频墙   | 多路 RTSP 实时播放      | ❌        | 需要 RTSP 输入 + WebRTC/HLS      |
| 实时控制     | 启动/停止检测、切换画面 | ❌        | UI 按钮是 Mock 模式              |
| 实时指标     | FPS/延迟/GPU 实时刷新   | ❌        | 离线 timeline FPS 有             |

## 六、已发现的具体代码问题


| # | 文件                           | 问题                                                        | 状态   |
| - | ------------------------------ | ----------------------------------------------------------- | ------ |
| 1 | `runtime_overrides.py:134-139` | `return` 后有无可达的重复代码                               | 待修复 |
| 2 | `gpu_monitor.py`               | 全文件占位，无`gpu_util()` 方法                             | 待实现 |
| 3 | `backpressure_controller.py`   | `mark_consumed()` 定义但零调用                              | 待修复 |
| 4 | `infer_primary_yolo.txt`       | 引用`yolov8n.engine` 但 `models/` 目录只有 `yolov8s.engine` | 待确认 |
| 5 | 多个配置文件                   | 硬编码`/home/nvidia/Desktop/YOLO/` 绝对路径                 | 待清理 |

---

## 七、总体差距汇总

```
AGX.md 最终目标:
  RTSP实时流 → 单Pipeline合批推理 → 实时跟踪 → C++热路径 →
  GPU自适应调度 → RTMP推流 → 实时Web监控 → Profiling体系

当前已达成:
  本地MP4 → 8独立进程 → 检测+跟踪 → JSONL输出 →
  批量验收+HTML报告 → 离线Web看板 → parser.so可用


🔴 架构层 (最大差距):
  1. 单 pipeline 多 source 合批 (替代 8 独立进程)
  2. RTSP 实时流接入 (替代本地 MP4)
  3. RTMP 实时推流 (替代本地 MP4 输出)

🔴 优化层:
  4. 丢帧策略切换：自适应概率丢帧 → 有界缓冲队列 (设计已完成，待实现)
  5. GPU 监控接入 tegrastats
  6. Profiling 体系建立

🟡 C++ 加速层 (设计已有, 代码未写):
  7. C++ Probe/Meta Parser 热路径下沉
  8. 轻量自定义 GStreamer 插件

🟡 增强层:
  9.  FP16 → INT8 量化
  10. IOU → NvDCF tracker
  11. MQTT/Kafka 输出
  12. Web 实时控制面 (当前为 Mock 模式)
  13. 长视频稳定性验证
  14. systemd 服务化部署
```

---

## 八、RTSP 入流 + RTMP 推流：MediaMTX 统一方案

> MediaMTX 同时支持 RTSP Server 和 RTMP Server，一个进程解决"模拟摄像头输入"和"推流输出 + 浏览器播放"两端需求。

### 8.1 整体架构

```
                     MediaMTX (单进程)
本地 MP4 ──RTSP──→  (端口 8554)  ──→ DeepStream Pipeline
                                      │  nvinfer → nvtracker → osd
                                      │  nvv4l2h264enc → flvmux → rtmpsink
DeepStream ───RTMP──→  (端口 1935)  ──→ MediaMTX → HLS (端口 8889) → 浏览器 <video>
```

### 8.2 安装

```bash
# Jetson AGX Orin (ARM64)
wget https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_arm64v8.tar.gz
tar xzf mediamtx_v1.9.3_linux_arm64v8.tar.gz
```

### 8.3 配置文件 `mediamtx.yml`

```yaml
# ──── RTSP 入流：6路本地 MP4 模拟摄像头 ────
paths:
  camera_01:
    source: ffmpeg -re -stream_loop -1 -i /home/nvidia/Desktop/YOLO/video/1.mp4 -c copy -f rtsp rtsp://localhost:$RTSP_PORT/camera_01
    sourceOnDemand: yes
  camera_02:
    source: ffmpeg -re -stream_loop -1 -i /home/nvidia/Desktop/YOLO/video/2.mp4 -c copy -f rtsp rtsp://localhost:$RTSP_PORT/camera_02
    sourceOnDemand: yes
  # ... 最多 camera_06

  # ──── RTMP 出流：DeepStream pipeline 推流目标 ────
  # 路径名 live/stream 与 builder.py 默认 location 一致
  # MediaMTX 自动将 RTMP 转为 HLS，浏览器可直接播放
```

### 8.4 启动

```bash
./mediamtx
```

### 8.5 项目配置对应

入流侧 — `streams.yaml`：

```yaml
sources:
  - name: camera_01
    kind: rtsp
    uri: rtsp://127.0.0.1:8554/camera_01
    enabled: true
  - name: camera_02
    kind: rtsp
    uri: rtsp://127.0.0.1:8554/camera_02
    enabled: true
```

出流侧 — `app.yaml` 切换到 RTMP 模式：

```yaml
deepstream:
  output_sink: rtmp        # file → rtmp 切换
  # output_video_path 不再需要（rtmp 模式忽略此字段）
```

builder.py 中 RTMP sink 默认目标（无需修改）：

```python
# builder.py:419 — 与 mediamtx.yml 中 path 对应
return {"location": "rtmp://127.0.0.1/live/stream"}
```

### 8.6 三种 Sink 模式

| 模式 | `output_sink` | Sink 元素 | 输出 | 当前使用 |
|------|--------------|----------|------|---------|
| 文件 | `file` | `filesink` | 本地 MP4 | ✅ 离线模式 |
| 推流 | `rtmp` | `rtmpsink` | MediaMTX → HLS → 浏览器 | 🟡 实时模式（待验证） |
| 丢弃 | `fake` | `fakesink` | 不输出视频 | 🟡 调试用 |

### 8.7 与 Web UI 的关系

**离线模式**（当前 `output_sink: file`）：

```
run_person_analytics_batch.sh → outputs/batch/
  → preview_web.py → Web UI 读 batch JSON + HTTP <video> 播本地 MP4
```

**实时模式**（将来 `output_sink: rtmp`）：

```
DeepStream → MediaMTX (RTMP 1935) → 自动转 HLS (8889)
  → <video src="http://jetson:8889/live/stream/index.m3u8">
```

两者互不冲突——离线模式用于批量验收，实时模式用于在线监控。`app.js` mock 数据只需改为真实 HLS URL 即可切换。

### 8.8 浏览器兼容性

浏览器不支持原生 RTMP，MediaMTX 自动将 RTMP 转为 HLS：

```html
<video src="http://jetson-ip:8889/live/stream/index.m3u8" controls></video>
```

不需要引入第三方播放器。

### 8.9 好处总结

- 不依赖公共 RTSP 测试流，同一批 MP4 文件直接复用
- `sourceOnDemand: yes` → pipeline 没连时不消耗 FFmpeg 解码资源
- 后续接入真实摄像头时，只换 RTSP URL，其他代码不变
- RTMP 出流也由 MediaMTX 承载，不用额外部署 nginx-rtmp


## 九、从 demo_multhread_decode_infer_mulmodel 可借鉴的设计模式

> 来源：Rockchip C++ 多路多模型推理项目 (`demo_multhread_decode_infer_mulmodel/`)。以下模式与 Jetson 项目的目标架构高度吻合，可直接指导缺口实现。

### 🔴 可直接借鉴（高价值 + 低迁移成本）


| # | 借鉴点                    | 来源文件                    | 核心思路                                                                                                    | 对应 Jetson 缺口                                       |
| - | ------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 1 | **断流重连 + 异常恢复**   | `stream_loader.cpp:322-385` | `operator()()` 主循环 try/catch；EOF 后 `close()` 清理资源；区分本地文件(立即reopen) vs RTSP(sleep 10s重试) | RTSP接入 +`handle_error()` 改造                        |
| 2 | **线程池 + 任务异常隔离** | `ThreadPool.hpp:193-242`    | worker 中`try { task(); } catch(...) { /* 不杀线程 */ }`；闲置线程超时自动回收                              | JSON 异步消费者线程 (`queue.Queue`)                    |
| 3 | **硬件加速 + 软件回退**   | `stream_loader.cpp:62-80`   | 优先 RGA 硬件 NV12→BGR，失败回退 OpenCV`cvtColor`，错误计数限刷屏                                          | `gpu_monitor` tegrastats→NVML→占位值三级回退         |
| 4 | **跳帧推理 (Smart Skip)** | `main.cpp:146-169`          | `INFER_INTERVAL=2`，偶数帧复用上一帧检测结果；`last_fused.empty()` 兜底                                     | `fps_controller` 增加确定性规则 (连续无检测→跳帧翻倍) |

### 🟡 架构思路可借鉴（高价值 + 需适配）


| # | 借鉴点                     | 来源文件                     | 核心思路                                                                                                   | 对应 Jetson 缺口                                          |
| - | -------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 5 | **单例流管理器**           | `stream_loader.h:84-131`     | `StreamLoaderManager` 单例持有 `vector<StreamLoader*>`，集中管理多路流的 load/unload/重连                  | 多 source 运行时管理 (`nvstreammux` request pad 动态增删) |
| 6 | **多模型融合架构**         | `detection_fusion_manager.h` | `FusionConfig` 可配置 (IoU/IoM阈值, 模型权重)；`fuseDetections()` → IoU匹配 → 加权融合bbox/置信度 → NMS | AGX.md PGIE+SGIE 级联推理的融合层                         |
| 7 | **本地文件帧率限速**       | `stream_loader.cpp:313-317`  | 检测本地文件源帧率 →`throttle=true` → `sleep(frame_interval_ms * 1.2)`                                   | 本地 MP4 解码过快导致 pipeline 堆积                       |
| 8 | **进度日志 (每 N 帧报告)** | `rknnPool.hpp:198`           | 每 100 帧输出一次 detection count                                                                          | pipeline 可观测性增强                                     |

### 🟢 不宜直接照搬的点


| Rockchip 做法                          | 为什么不适用于 Jetson 项目                       |
| -------------------------------------- | ------------------------------------------------ |
| FFmpeg 软解 + MPP 硬解混用             | DeepStream 已用 NVDEC 统一硬解，不需要 FFmpeg    |
| `rknn_set_core_mask` 手动绑定 NPU 核心 | TensorRT 不需要手动指定 GPU core                 |
| `exit(-1)` 初始化失败直接退出          | 项目已有`raise` + 日志，应改为有序关闭 + restart |

---

## 十、讨论记录

### 2026-07-09 — demo 项目可借鉴设计模式分析

- 通读 `demo_multhread_decode_infer_mulmodel/` (Rockchip C++ 项目) 全部源码
- 识别出 8 个可借鉴设计模式：断流重连/异常恢复、线程池+任务异常隔离、硬件加速+软件回退、跳帧推理、单例流管理器、多模型融合、帧率限速、进度日志
- 识别出 3 个不宜直接照搬的做法 (FFmpeg混用、手动NPU绑定、exit硬退出)
- 已写入「八、从 demo_multhread_decode_infer_mulmodel 可借鉴的设计模式」

### 2026-07-09 — 丢帧策略设计决策：有界缓冲队列替代自适应概率丢帧

- **决策**：用 `queue.Queue(maxsize=N)` + `put_nowait()` 替代 `fps_controller` 的概率丢帧 + `backpressure_controller` 的生产消费对账
- **理由**：自适应方案依赖 GPU 指标（当前占位值），反应式缓冲队列无外部依赖、确定性触发、天然吸收突发
- **影响**：`fps_controller.py` 和 `backpressure_controller.py` 降级为监控辅助模块，丢帧决策交给 `queue.Queue` 本身
- **参考**：Rockchip demo 项目的 `Mbuffer`（共享 `images[]` + `mutexes[]`）是同一模式
- 已写入「四、4.1 丢帧策略设计决策」

### 2026-07-09 — RTSP 模拟方案 + RTMP 推流分析

- **RTSP 模拟**：确定采用 MediaMTX 将本地 MP4 发布为 RTSP 流，`sourceOnDemand: yes` 按需启动。6 路文件通过 FFmpeg 循环推送到 MediaMTX，pipeline 拉 `rtsp://127.0.0.1:8554/camera_0X`
- **RTMP 推流原理**：pipeline 编码链路为 `nvv4l2h264enc → h264parse → flvmux → rtmpsink`，推送到 RTMP 服务器（MediaMTX 端口 1935）
- **与 UI 不冲突**：离线模式 UI 读 batch JSON + 播本地 MP4；实时模式 UI 通过 MediaMTX 转 HLS（端口 8889），`<video>` 直接播放。两种模式不同时运行
- **统一方案**：MediaMTX 一个进程同时承载 RTSP 入流 + RTMP 出流 + HLS 转码
- 已写入「八、RTSP 模拟方案」和「九、RTMP 推流」

---

## 十一、更新规则

本文档在以下情况需要更新：

1. 每完成一项差距任务 → 更新对应行状态
2. 发现新的代码问题 → 追加到「具体代码问题」表格
3. 讨论产生新的架构决策 → 追加到「讨论记录」
4. 优先级重新评估 → 更新「总体差距汇总」排序

每次更新后，刷新文末「最后更新」日期。
