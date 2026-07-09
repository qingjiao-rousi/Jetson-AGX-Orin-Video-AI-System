# YOLOv8 Person Detection 最小运行指南

本文档说明当前项目的最小可运行版本如何使用、如何验证、输入输出是什么，以及常见问题如何排查。当前目标不是完整多路视频平台，而是先稳定完成一个闭环：

```text
本地 MP4 输入 -> DeepStream YOLOv8 person 检测 -> 带框 MP4 输出 + JSONL 检测结果输出
```

## 1. 当前完成状态

当前最小实现已经完成以下能力：

- 读取本地 MP4 文件。
- 使用 DeepStream 7.1 管线执行 YOLOv8s 推理。
- 使用 TensorRT engine 加速推理。
- 使用 DeepStream-Yolo 自定义 parser 解析 YOLOv8 输出。
- 仅保留 `person` 类别，过滤 COCO 其他类别。
- 使用 `nvdsosd` 在视频上绘制检测框。
- 使用 Jetson 硬件 H.264 编码器输出 MP4。
- 将每帧检测结果写入 JSONL。
- 启用 DeepStream tracker 后输出 `track_id`。
- 视频框标签显示 `person ID:<track_id>`。
- 提供一键运行脚本和一键验收脚本。

关键输出文件：

```text
outputs/person_detect.mp4
outputs/results.jsonl
```

## 2. 项目目录说明

常用目录和文件如下：

```text
configs/app/app_minimal.yaml
```

最小运行配置。包含输入源数量、输出视频路径、是否启用 OSD、输出尺寸等默认值。

```text
configs/deepstream/infer_primary_yolo_minimal.txt
```

DeepStream `nvinfer` 配置。包含 engine、ONNX、labels、自定义 parser、person-only 过滤、阈值等。

```text
models/yolov8s.onnx
models/yolov8s.engine
models/labels.txt
```

YOLOv8s 模型资源。当前 `yolov8s.onnx` 应使用 DeepStream-Yolo 的 `export_yoloV8.py` 导出，否则 parser 可能解析出错误类别、错误 bbox 或异常 confidence。

```text
custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

DeepStream-Yolo 自定义解析库。

```text
scripts/run_person_detect.sh
```

最小工具运行入口。

```text
scripts/check_person_output.sh
```

输出验收脚本。

```text
scripts/summarize_person_tracks.py
```

离线统计脚本，读取 tracker JSONL 并生成 `summary.json`。

```text
scripts/summarize_person_roi.py
```

离线 ROI 统计脚本，使用 `track_id + bbox 中心点` 判断人员是否在区域内。

```text
scripts/summarize_person_line.py
```

离线越线计数脚本，使用 `track_id + bbox 中心点轨迹` 判断是否穿过一条线。

```text
configs/analytics/person_analytics.yaml
```

统一 analytics 配置文件，集中定义 ROI 和计数线。

```text
scripts/summarize_person_analytics.py
```

统一 analytics summary 脚本，一次输出全局计数、ROI 统计、越线统计。

```text
scripts/draw_person_analytics.py
```

可视化叠加脚本，把 ROI 矩形和计数线画到输出视频上。

```text
scripts/summarize_person_timeline.py
```

离线时间轴与同步基准脚本，检查 `stream_id/frame_id/timestamp` 的连续性、单调性和估算 FPS。

## 3. 环境要求

目标环境：

- Jetson AGX Orin 32G
- Ubuntu / JetPack 对应系统
- DeepStream 7.1
- TensorRT
- CUDA
- GStreamer
- Python 3.10
- `pyds`
- DeepStream-Yolo parser `.so`

项目默认 DeepStream 路径：

```text
/opt/nvidia/deepstream/deepstream-7.1
```

加载项目环境变量：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master
source scripts/env.sh
```

检查基础环境：

```bash
scripts/check_env.sh
```

`libtritonserver.so`、`librivermax.so.0` 相关 warning 通常不影响当前最小 YOLO 文件检测流程，因为当前不使用 Triton Inference Server 和 Rivermax UDP 插件。

## 4. 推荐运行方式

### 4.1 最终验收入口

当前接近落地版本推荐使用一条命令完成：

```text
8 路本地 MP4 并行处理 -> 输出视频/JSONL/summary/quality -> 启动本地 UI 展示结果
```

运行：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master
source scripts/env.sh

scripts/run_acceptance_ui.sh
```

默认输入：

```text
/home/nvidia/Desktop/YOLO/video
```

默认输出：

```text
outputs/acceptance_latest
```

默认 UI：

```text
http://127.0.0.1:8090
```

该命令会先执行默认 8 路并行批量 analytics，然后自动启动本地 Web UI。浏览器打开上面的地址后，可以在“批量视频结果看板”里查看每路视频的 overlay 播放、人数、ROI、越线、FPS、质量状态和错误/复核原因。

UI 批量看板包含：

- 验收结论：通过、需复核或失败。
- 筛选：全部、复核、失败、有人、无人、FPS 异常、帧不连续。
- 单视频详情：人数、ROI、越线、时间轴 FPS、处理 FPS、总帧数、耗时、文件大小和 `run.log` 摘要。
- 一键打开：`batch_summary.json`、`batch_quality.json`、HTML 报告、CSV 报告。
- 单视频输出入口：播放视频、overlay 文件、JSONL、summary、run.log。
- 性能基准：总耗时、`BATCH_JOBS` 并发数和处理 FPS。

注意：UI 的主播放器默认播放 `person_analytics.mp4`，这是 DeepStream 硬件编码输出的 H.264 MP4，浏览器兼容性更好。`person_analytics_overlay.mp4` 由 OpenCV 离线绘制 ROI/line 后生成，部分浏览器可能不支持其编码，页面会保留 `overlay文件` 链接用于下载或外部播放器查看。

离线验收模式下，页面会优先展示本次批量结果、质量状态、时间轴 FPS 和处理 FPS。旧的实时视频墙、Pipeline 日志和原始调试快照已经折叠到“高级实时调试区”，避免干扰当前离线批量验收主流程。实时 GPU/资源监控后续在服务化/实时流阶段再接入。

两个 FPS 的含义不同：

- 时间轴 FPS：由输出 JSONL 的时间轴估算，主要反映视频帧时间戳和离线结果时间线。
- 处理 FPS：`总帧数 / 实际运行耗时`，更适合判断当前机器处理吞吐，以及推理、解码、编码或磁盘写入是否成为瓶颈。

如果只想跑验收，不启动 UI：

```bash
START_UI=0 scripts/run_acceptance_ui.sh
```

如果要指定输入和输出目录：

```bash
scripts/run_acceptance_ui.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/acceptance_run_001
```

如果要临时使用 4 路并行：

```bash
BATCH_JOBS=4 scripts/run_acceptance_ui.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/acceptance_4
```

### 4.2 单视频运行

推荐使用最终统一入口：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master

scripts/run_person_analytics.sh \
  /home/nvidia/Desktop/YOLO/video/1.mp4 \
  outputs/final
```

默认输出：

```text
outputs/final/person_analytics.mp4
outputs/final/results.jsonl
outputs/final/analytics_summary.json
outputs/final/person_analytics_overlay.mp4
```

这条命令会完成：

1. person 检测。
2. tracker 跟踪。
3. 带 `ID + confidence` 的 MP4 输出。
4. JSONL 输出。
5. 输出检查。
6. 统一 analytics summary。
7. ROI/line 可视化叠加视频。

可选参数通过环境变量覆盖：

```bash
OUTPUT_WIDTH=1280 OUTPUT_HEIGHT=720 CONFIDENCE_THRESHOLD=0.35 \
ANALYTICS_CONFIG=configs/analytics/person_analytics.yaml \
scripts/run_person_analytics.sh \
  /home/nvidia/Desktop/YOLO/video/1.mp4 \
  outputs/final
```

如果只想生成 JSON 和 summary，不生成 overlay 视频：

```bash
SKIP_OVERLAY=1 scripts/run_person_analytics.sh \
  /home/nvidia/Desktop/YOLO/video/1.mp4 \
  outputs/final
```

如果要跳过输出检查：

```bash
SKIP_CHECK=1 scripts/run_person_analytics.sh \
  /home/nvidia/Desktop/YOLO/video/1.mp4 \
  outputs/final
```

## 5. 基础检测运行方式

使用默认输出路径：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master

scripts/run_person_detect.sh /home/nvidia/Desktop/YOLO/video/1.mp4
```

运行完成后会输出：

```text
outputs/person_detect.mp4
outputs/results.jsonl
```

指定输入和输出：

```bash
scripts/run_person_detect.sh \
  /home/nvidia/Desktop/YOLO/video/1.mp4 \
  outputs/person_detect.mp4 \
  outputs/results.jsonl
```

调整输出尺寸和置信度阈值：

```bash
OUTPUT_WIDTH=1280 OUTPUT_HEIGHT=720 CONFIDENCE_THRESHOLD=0.35 \
scripts/run_person_detect.sh \
  /home/nvidia/Desktop/YOLO/video/1.mp4 \
  outputs/person_detect.mp4 \
  outputs/results.jsonl
```

说明：

- `OUTPUT_WIDTH` / `OUTPUT_HEIGHT` 控制最终输出视频尺寸。
- `CONFIDENCE_THRESHOLD` 控制 `nvinfer` 的 `pre-cluster-threshold`。
- 默认只保留 `person` 类别。
- 批处理脚本默认关闭 Web dashboard，避免占用或绑定 `8080` 端口。需要 dashboard 时可加 `ENABLE_WEB=1`。

开启 dashboard：

```bash
ENABLE_WEB=1 scripts/run_person_detect.sh /home/nvidia/Desktop/YOLO/video/1.mp4
```

然后访问：

```text
http://127.0.0.1:8080
```

## 6. 统一验收入口

早期分阶段 `smoke_test_person_*.sh` 脚本已经删除，当前推荐统一使用最终验收入口：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master

scripts/run_acceptance_ui.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/acceptance_latest
```

如果只验收单个视频：

```bash
scripts/run_person_analytics.sh \
  /home/nvidia/Desktop/YOLO/video/1.mp4 \
  outputs/final
```

如果验收单进程单 pipeline 多路合批：

```bash
scripts/run_multifile_inproc.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/multifile_inproc
```

## 7. 手动验收命令

如果不使用烟测脚本，可以手动验收。

检查输出文件：

```bash
ls -lh outputs/person_detect.mp4 outputs/results.jsonl
```

检查视频是否可解析：

```bash
gst-discoverer-1.0 outputs/person_detect.mp4
```

正常情况下应看到类似：

```text
container #0: Quicktime
  video #1: H.264
    Width: 1280
    Height: 720
    Frame rate: 50/1
```

查看 JSONL 中的检测结果：

```bash
grep -F '"detections": [{' outputs/results.jsonl | head -3
```

正常检测项应类似：

```json
{
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.82,
  "bbox": {
    "left": 100.0,
    "top": 80.0,
    "width": 60.0,
    "height": 180.0
  }
}
```

重点检查：

- `class_id` 应为 `0`。
- `class_name` 应为 `person`。
- `confidence` 应在 `0..1`。
- `bbox.width` 和 `bbox.height` 应大于 `0`。

## 8. Tracker 验收

当前最小配置已启用 tracker：

```yaml
enable_tracker: true
```

tracker 的目标是给跨帧出现的同一个人分配稳定 ID。视频框标签会显示类似：

```text
person ID:5
```

JSONL 中会出现：

```json
"tracks": [
  {
    "track_id": 5,
    "class_id": 0,
    "bbox": {
      "left": 120.0,
      "top": 88.0,
      "width": 52.0,
      "height": 170.0
    }
  }
]
```

手动检查 tracker 输出：

```bash
REQUIRE_TRACKS=1 scripts/check_person_output.sh \
  outputs/smoke/person_tracker.mp4 \
  outputs/smoke/tracker_results.jsonl
```

验收标准：

- MP4 可以播放。
- JSONL 非空。
- detections 仍然只包含 `person`。
- `tracks` 非空。
- `track_id` 存在。
- 至少一个 `track_id` 跨多帧出现。

如果 tracker 报缺少 `libcufft.so` 或 `libcublas.so`，先确认环境变量：

```bash
source scripts/env.sh
ldd /opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so | grep "not found"
```

正常情况下该命令不应输出任何内容。项目已在 `scripts/env.sh` 中加入 CUDA 12.6 target lib 路径：

```text
/usr/local/cuda-12.6/targets/aarch64-linux/lib
```

## 9. 去重计数统计

tracker 通过后，可以基于 `track_id` 做离线去重统计。统计脚本不依赖 DeepStream，只读取 JSONL：

```bash
python3 scripts/summarize_person_tracks.py \
  outputs/smoke/tracker_results.jsonl \
  outputs/smoke/summary.json
```

输出示例：

```json
{
  "total_unique_persons": 3,
  "total_frames": 286,
  "frames_with_person": 282,
  "max_persons_in_frame": 3,
  "stable_track_ids": [0, 1, 2],
  "tracks": [
    {
      "track_id": 0,
      "first_frame": 4,
      "last_frame": 285,
      "frame_count": 282
    }
  ]
}
```

字段说明：

- `total_unique_persons`：按稳定 `track_id` 去重后的人数。
- `total_frames`：JSONL 总帧数。
- `frames_with_person`：有检测到人的帧数。
- `frames_with_tracks`：有跟踪结果的帧数。
- `max_persons_in_frame`：单帧最大检测人数。
- `max_tracks_in_frame`：单帧最大跟踪人数。
- `stable_track_ids`：满足最小出现帧数的 track ID。
- `tracks`：每个稳定 track 的首帧、尾帧、出现帧数和最后 bbox。

默认至少出现 2 帧才计为稳定 track。可以调整：

```bash
python3 scripts/summarize_person_tracks.py \
  outputs/smoke/tracker_results.jsonl \
  outputs/smoke/summary.json \
  --min-track-frames 5
```

## 10. ROI 区域人数统计

ROI 统计基于 tracker JSONL 中的 `tracks`，用每个 track 的 bbox 中心点判断是否落入矩形区域。

bbox 中心点计算：

```text
center_x = bbox.left + bbox.width / 2
center_y = bbox.top + bbox.height / 2
```

运行 ROI 统计：

```bash
python3 scripts/summarize_person_roi.py \
  outputs/smoke/tracker_results.jsonl \
  outputs/smoke/roi_summary.json \
  --roi 0,0,1280,720 \
  --roi-id full-frame
```

`--roi` 格式：

```text
x,y,width,height
```

例如只统计画面左半边：

```bash
python3 scripts/summarize_person_roi.py \
  outputs/smoke/tracker_results.jsonl \
  outputs/smoke/roi_left.json \
  --roi 0,0,640,720 \
  --roi-id left-half
```

ROI summary 示例：

```json
{
  "roi_id": "full-frame",
  "roi": {
    "x": 0.0,
    "y": 0.0,
    "width": 1280.0,
    "height": 720.0
  },
  "unique_persons_in_roi": 3,
  "frames_with_roi_person": 282,
  "max_persons_in_roi_frame": 3,
  "stable_track_ids": [0, 1, 2],
  "tracks": [
    {
      "track_id": 0,
      "first_frame": 4,
      "last_frame": 285,
      "frames_in_roi": 282,
      "average_confidence": 0.82,
      "max_confidence": 0.91
    }
  ]
}
```

### 9.1 置信度在哪一步产生

置信度由 YOLOv8 模型在 `nvinfer` 推理后产生，再由 DeepStream-Yolo 的 `NvDsInferParseYolo` parser 解析到 `NvDsObjectMeta.confidence`。

当前项目会将 confidence 写入两个位置：

```json
"detections": [
  {
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.86,
    "bbox": {}
  }
]
```

以及 tracker 输出：

```json
"tracks": [
  {
    "track_id": 5,
    "class_id": 0,
    "confidence": 0.86,
    "bbox": {}
  }
]
```

ROI summary 中的：

- `average_confidence`
- `max_confidence`

来自同一个 `track_id` 在 ROI 内多帧 confidence 的统计。

如果旧 JSONL 是在 track confidence 加入前生成的，那么 ROI summary 里的 confidence 可能是 `0.0`。重新运行 tracker 或 ROI smoke test 后，新 JSONL 会包含 track confidence。

## 11. 越线计数统计

越线计数基于每个 `track_id` 的 bbox 中心点轨迹。给定一条有向线段：

```text
x1,y1,x2,y2
```

脚本会比较同一个 track 上一帧和当前帧在线段两侧的位置。如果侧别发生变化，就记为一次跨线。

运行越线统计：

```bash
python3 scripts/summarize_person_line.py \
  outputs/smoke/tracker_results.jsonl \
  outputs/smoke/line_summary.json \
  --line 640,0,640,720 \
  --line-id middle-vertical
```

输出示例：

```json
{
  "line_id": "middle-vertical",
  "line_crossing_in": 1,
  "line_crossing_out": 0,
  "in_track_ids": [1],
  "out_track_ids": [],
  "crossing_count": 1
}
```

方向说明：

- 线段是有方向的，从 `(x1, y1)` 指向 `(x2, y2)`。
- `in/out` 根据中心点从线的一侧移动到另一侧判断。
- 对默认竖线 `640,0,640,720`，它是从上到下的线。不同方向定义可以通过交换两个端点来反转。

为了避免人在计数线附近抖动导致重复计数，默认每个 `track_id` 只计第一次跨线。

如果你确实希望同一个 track 多次跨线都计数：

```bash
python3 scripts/summarize_person_line.py \
  outputs/smoke/tracker_results.jsonl \
  outputs/smoke/line_summary.json \
  --line 640,0,640,720 \
  --no-count-once-per-track
```

可以通过 `--min-side-distance` 忽略贴近线附近的小抖动。

## 12. 统一 Analytics 配置与可视化

当前 ROI 和 line 可以统一写在：

```text
configs/analytics/person_analytics.yaml
```

示例：

```yaml
min_track_frames: 2

rois:
  - id: full-frame
    rect: [0, 0, 1280, 720]

lines:
  - id: middle-vertical
    points: [640, 0, 640, 720]
    min_side_distance: 1.0
    count_once_per_track: true
```

生成统一 summary：

```bash
python3 scripts/summarize_person_analytics.py \
  outputs/smoke/tracker_results.jsonl \
  configs/analytics/person_analytics.yaml \
  outputs/smoke/analytics_summary.json
```

输出包含：

- `timeline`：按 stream 汇总 frame_id 连续性、timestamp 单调性和估算 FPS。
- `global`：全视频去重人数和 track 列表。
- `rois`：每个 ROI 内的去重人数、帧数、置信度统计。
- `lines`：每条计数线的 `in/out` 计数和 crossing events。

把 ROI 和 line 画到视频上：

```bash
python3 scripts/draw_person_analytics.py \
  outputs/smoke/person_tracker.mp4 \
  configs/analytics/person_analytics.yaml \
  outputs/smoke/person_analytics_overlay.mp4
```

正式运行入口：

```bash
scripts/run_person_analytics.sh /home/nvidia/Desktop/YOLO/video/1.mp4 outputs/final
```

默认输出：

```text
outputs/smoke/person_analytics.mp4
outputs/smoke/analytics_results.jsonl
outputs/smoke/analytics_summary.json
outputs/smoke/person_analytics_overlay.mp4
```

### 12.1 视频中的置信度

视频框标签现在显示：

```text
person ID:<track_id> <confidence>
```

例如：

```text
person ID:5 0.86
```

其中 confidence 来自 `NvDsObjectMeta.confidence`，由 `nvinfer + NvDsInferParseYolo` 产生。

## 13. 离线时间轴与同步基准

时间轴脚本用于检查当前 JSONL 结果中的：

- `stream_id`
- `frame_id`
- `timestamp`

它会按 stream 分组，输出每路视频的帧连续性和 timestamp 单调性。

运行：

```bash
python3 scripts/summarize_person_timeline.py \
  outputs/smoke/tracker_results.jsonl \
  outputs/smoke/timeline_summary.json
```

输出示例：

```json
{
  "stream_count": 1,
  "streams": {
    "stream-0": {
      "frame_count": 286,
      "first_frame": 0,
      "last_frame": 285,
      "missing_frame_count": 0,
      "duplicate_frame_count": 0,
      "is_frame_continuous": true,
      "is_timestamp_monotonic": true,
      "duration_seconds": 3.32209,
      "estimated_fps": 85.78
    }
  }
}
```

字段说明：

- `is_frame_continuous`：`frame_id` 是否连续、无缺失、无重复、无倒序。
- `is_timestamp_monotonic`：timestamp 是否单调递增。
- `missing_frames`：缺失帧列表，最多输出前 100 个。
- `duplicate_frames`：重复帧列表，最多输出前 100 个。
- `out_of_order_frames`：倒序帧位置，最多输出前 100 个。
- `estimated_fps`：根据 JSONL timestamp 估算的结果时间轴 FPS。

注意：当前 timestamp 来自 DeepStream/Python 处理结果时间，而不一定等于原始视频 PTS。它适合作为当前离线处理链路的同步基准。后续接 RTSP、多路摄像头时，需要进一步区分：

- 源视频 PTS
- 摄像头 NTP
- DeepStream batch timestamp
- JSONL 写出时间

统一 analytics summary 已经包含 `timeline`：

```bash
python3 scripts/summarize_person_analytics.py \
  outputs/smoke/tracker_results.jsonl \
  configs/analytics/person_analytics.yaml \
  outputs/smoke/analytics_summary.json
```

## 14. 批量视频处理

如果 `/home/nvidia/Desktop/YOLO/video` 中有多个 MP4，可以使用批量入口处理。当前批量模式是 **离线多文件批处理**：

```text
多个本地 MP4 -> 调用单视频 analytics -> 每个视频独立输出 -> 汇总 batch_summary.json
```

它不是实时多路摄像头同时拉流，也不是 DeepStream 单 pipeline 多 source 合批处理。它可以通过 `BATCH_JOBS` 做 **受控并行**：同时启动多个独立的单视频 analytics 进程，每个视频写入自己的输出目录。

运行八个视频。当前落地版默认使用 8 路并行：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master

scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

更推荐使用最终验收入口，它会跑完批量后直接启动 UI：

```bash
scripts/run_acceptance_ui.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/acceptance_latest
```

等价于：

```bash
BATCH_JOBS=8 scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

如果需要回退到最稳的串行模式：

```bash
BATCH_JOBS=1 scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

如果要使用 4 路并行做瓶颈对比：

```bash
BATCH_JOBS=4 scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

当前先保留 8 路并行作为项目接近落地版本的默认策略，后续再围绕瓶颈做优化。多个进程会同时争用：

- GPU/TensorRT
- NVDEC/NVENC
- 内存带宽
- 磁盘写入

后续瓶颈优化时建议对比：

```text
BATCH_JOBS=1 -> 串行基准
BATCH_JOBS=4 -> 常用稳定并发点
BATCH_JOBS=8 -> 当前落地默认策略
```

### 14.1 单进程多路合批实验入口

当前已经新增一个单进程多路实验入口，用于验证：

```text
8 个本地 MP4
-> 1 个 Python/DeepStream 进程
-> 1 个 nvstreammux batch-size=8
-> nvinfer batch 推理
-> tracker / osd probe
-> nvmultistreamtiler 2x4 拼接
-> 1 个 tiled MP4 + 1 个合并 JSONL
```

运行：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master
source scripts/env.sh

scripts/run_multifile_inproc.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/multifile_inproc
```

默认输出：

```text
outputs/multifile_inproc/results.jsonl
outputs/multifile_inproc/multifile_preview.mp4
outputs/multifile_inproc/multifile_summary.json
outputs/multifile_inproc/multifile_quality.json
outputs/multifile_inproc/run.log
outputs/multifile_inproc/.runtime/app_multifile_runtime.yaml
```

该入口默认使用：

```text
OUTPUT_SINK=file
ENABLE_TILER=1
TILER_ROWS=2
TILER_COLUMNS=4
TILER_WIDTH=1280
TILER_HEIGHT=720
```

也就是说当前单 pipeline 会输出一个 2x4 拼接预览视频，同时输出合并 JSONL。JSONL 中会通过 `stream_id/source_id` 区分每一路输入。

运行脚本结束后会自动生成：

- `multifile_summary.json`：按 `stream_id` 汇总帧数、检测数、track 观测、去重人数、时间轴 FPS 和帧连续性。
- `multifile_quality.json`：检查 tiled MP4、JSONL、run.log 是否存在，8 路 stream 是否齐全，并标记 passed/review/failed。

本地 UI 已增加“单 Pipeline 结果看板”，会展示：

- `multifile_preview.mp4` tiled 视频。
- 8 路 stream 统计表。
- 单 pipeline 质量状态。
- summary / quality / JSONL / run.log 快捷入口。

如果单独启动 UI 预览已有结果：

```bash
python3 scripts/preview_web.py \
  --host 127.0.0.1 \
  --port 8090 \
  --batch-dir outputs/acceptance_latest \
  --multifile-dir outputs/multifile_inproc
```

如果只想先测 4 路：

```bash
SOURCE_COUNT=4 scripts/run_multifile_inproc.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/multifile_4
```

如果单进程多路 tiled MP4 和 JSONL 跑通，下一步再选择是否需要分路输出：

- 保持合成一路预览：继续使用 `nvmultistreamtiler`，适合 UI 总览和验收演示。
- 每路单独输出：在推理/OSD 后增加 `nvstreamdemux`，每路接独立 encoder/sink。

### 14.2 用本地 MP4 模拟 RTSP 拉流

如果当前没有真实 RTSP 摄像头，可以先用本地 MP4 启动一个本地 RTSP 模拟器：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master
source scripts/env.sh

python3 scripts/serve_mp4_as_rtsp.py \
  /home/nvidia/Desktop/YOLO/video \
  --limit 8 \
  --port 8554
```

它会暴露：

```text
rtsp://127.0.0.1:8554/stream1
rtsp://127.0.0.1:8554/stream2
...
rtsp://127.0.0.1:8554/stream8
```

然后在另一个终端运行单进程 RTSP 拉流入口：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master
source scripts/env.sh

scripts/run_rtsp_inproc.sh outputs/rtsp_inproc
```

默认输出：

```text
outputs/rtsp_inproc/results.jsonl
outputs/rtsp_inproc/run.log
outputs/rtsp_inproc/.runtime/app_rtsp_runtime.yaml
```

注意：

- MP4 转 RTSP 是为了模拟真实摄像头的 live-source、网络拉流、延迟和动态 pad。
- 如果只是为了验证单进程多路合批，优先使用 `scripts/run_multifile_inproc.sh`，不必先转 RTSP。
- RTMP 推流需要本机或局域网中存在 RTMP server，例如 nginx-rtmp 或 MediaMTX。当前优先验证 RTSP 拉流和单进程合批，RTMP 输出放到后续视频输出阶段处理。

默认匹配：

```text
*.mp4
```

输出结构：

```text
outputs/batch/
  001_video_a/
    person_analytics.mp4
    results.jsonl
    analytics_summary.json
    person_analytics_overlay.mp4
    run_metadata.json
    run.log
    .runtime/
  002_video_b/
    person_analytics.mp4
    results.jsonl
    analytics_summary.json
    person_analytics_overlay.mp4
    run_metadata.json
  batch_summary.json
  batch_summary.csv
  batch_report.html
  batch_quality.json
```

每个子目录对应一个输入视频。`run_metadata.json` 记录该视频的输入路径、开始时间、结束时间、退出码和运行状态。

并行模式下每个视频还会写入：

- `run.log`：该视频完整运行日志，方便定位单个视频失败原因。
- `.runtime/`：该视频独立的 DeepStream runtime config，避免多个并行进程同时覆盖同一个配置文件。

`batch_summary.json` 汇总所有视频，核心字段包括：

- `video_count`：批量目录中实际处理的视频数量。
- `processed_count`：成功处理数量。
- `failed_count`：失败数量。
- `total_unique_persons_sum`：各视频去重人数之和。
- `line_crossing_in_sum` / `line_crossing_out_sum`：所有视频越线 in/out 汇总。
- `videos`：每个视频的输入路径、输出路径、ROI 统计、越线统计和时间轴摘要。

`batch_summary.csv` 是表格版结果，适合用 Excel/WPS 打开进行人工复核。

`batch_report.html` 是本地报告页面，包含：

- 总览统计。
- 每个视频的处理状态。
- 每个视频的去重人数、ROI 人数、越线 in/out。
- 每个视频的帧数、估算 FPS、帧连续性。
- 可点击的 `video`、`overlay`、`jsonl`、`summary` 输出链接。
- 失败视频的错误原因。

`batch_quality.json` 是批量质量检查结果，用来标记哪些视频可以直接通过、哪些需要人工复核、哪些失败。

质量状态分三类：

- `passed`：输出文件完整，运行成功，时间轴基本正常。
- `review`：运行成功但存在可疑项，例如无人、无时间轴、帧不连续、FPS 异常。
- `failed`：运行失败，或关键输出文件缺失/为空。

查看汇总：

```bash
python3 -m json.tool outputs/batch/batch_summary.json | less
```

手动重新导出 CSV/HTML 报告：

```bash
python3 scripts/export_person_batch_report.py \
  outputs/batch/batch_summary.json \
  outputs/batch
```

生成：

```text
outputs/batch/batch_summary.csv
outputs/batch/batch_report.html
```

手动重新执行批量质量检查：

```bash
python3 scripts/check_person_batch_outputs.py \
  outputs/batch/batch_summary.json \
  outputs/batch/batch_quality.json
```

如果业务要求每个视频都必须检测到人，可以加：

```bash
python3 scripts/check_person_batch_outputs.py \
  outputs/batch/batch_summary.json \
  outputs/batch/batch_quality.json \
  --require-person
```

在桌面环境中可以直接打开 HTML 报告：

```bash
xdg-open outputs/batch/batch_report.html
```

也可以启动本地 Web UI 查看批量结果：

```bash
python3 scripts/preview_web.py \
  --host 127.0.0.1 \
  --port 8090 \
  --batch-dir outputs/batch
```

然后浏览器访问：

```text
http://127.0.0.1:8090
```

页面中的“批量视频结果看板”会读取：

```text
outputs/batch/batch_summary.json
outputs/batch/batch_quality.json
```

点击表格中的任意视频行，会在右侧播放对应的 `person_analytics_overlay.mp4`，并显示该视频的人数、ROI、越线、FPS、质量状态和错误/复核原因。

只处理特定文件名模式：

```bash
VIDEO_GLOB='*.MP4' scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

如果某个视频失败，默认继续处理后续视频：

```bash
CONTINUE_ON_ERROR=1 scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

如果希望遇到第一个失败就停止：

```bash
CONTINUE_ON_ERROR=0 scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

常用覆盖参数：

```bash
OUTPUT_WIDTH=1280 OUTPUT_HEIGHT=720 CONFIDENCE_THRESHOLD=0.35 \
ANALYTICS_CONFIG=configs/analytics/person_analytics.yaml \
scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

如果只需要 JSONL 和 summary，不生成 ROI/line overlay 视频：

```bash
SKIP_OVERLAY=1 scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

如果暂时跳过输出验收：

```bash
SKIP_CHECK=1 scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

批量处理建议先在 Jetson 本机终端运行，因为它会调用 DeepStream、TensorRT、硬件解码和硬件编码。Codex 沙盒环境通常没有完整 GPU/NVMM 设备访问权限，适合跑单元测试，不适合做真实视频推理验收。

## 15. 输出说明

### 15.1 MP4 视频输出

输出视频路径由脚本第二个参数控制：

```bash
scripts/run_person_detect.sh INPUT_MP4 OUTPUT_MP4 OUTPUT_JSONL
```

例如：

```text
outputs/person_detect.mp4
```

当前输出链路使用 Jetson 硬件 H.264 编码：

```text
nvvideoconvert
-> video/x-raw(memory:NVMM),format=NV12
-> nvv4l2h264enc
-> h264parse
-> qtmux
-> filesink
```

其中 `nvv4l2h264enc` 是 Jetson 上的 NVIDIA V4L2 硬件编码器。运行日志中出现以下内容，说明硬件编码器被调用：

```text
===== NvVideo: NVENC =====
H264: Profile = 66 Level = 0
NVMEDIA: Need to set EMC bandwidth
```

### 15.2 JSONL 输出

JSONL 路径由脚本第三个参数控制：

```text
outputs/results.jsonl
```

每一行是一帧结果，典型结构：

```json
{
  "stream_id": "stream-0",
  "frame_id": 123,
  "timestamp": "2026-07-06 15:03:59.254579+00:00",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.86,
      "bbox": {
        "left": 120.0,
        "top": 88.0,
        "width": 52.0,
        "height": 170.0
      }
    }
  ],
  "tracks": [],
  "extra": {}
}
```

tracker 启用后，`tracks` 中会写入 `track_id`。如果某一帧没有稳定跟踪对象，`tracks` 可能为空。

## 16. 当前最小管线

本地 MP4 输入链路：

```text
filesrc
-> qtdemux
-> queue
-> h264parse
-> nvv4l2decoder
-> nvvideoconvert
-> video/x-raw(memory:NVMM),format=NV12
-> queue
-> nvstreammux
```

推理和输出链路：

```text
nvstreammux
-> nvinfer
-> nvtracker
-> nvvideoconvert
-> video/x-raw(memory:NVMM),format=RGBA
-> nvdsosd
-> queue
-> nvvideoconvert
-> video/x-raw(memory:NVMM),format=NV12
-> nvv4l2h264enc
-> h264parse
-> qtmux
-> filesink
```

metadata 输出：

```text
osd sink pad probe
-> pyds.gst_buffer_get_nvds_batch_meta
-> NvDsFrameMeta / NvDsObjectMeta
-> JSONL
```

## 17. 常见问题

### 17.1 `outputs/person_detect.mp4` 只有 1.5K，无法播放

这通常说明编码器打开了，但没有收到有效帧，或者 MP4 mux 没有正确收尾。

当前项目已修复一个关键问题：Python pad probe 必须返回 `Gst.PadProbeReturn.OK`，不能返回 `0`。返回 `0` 会触发 `GST_PAD_PROBE_DROP`，导致后续帧被丢弃。

重新运行：

```bash
scripts/run_person_detect.sh /home/nvidia/Desktop/YOLO/video/1.mp4
scripts/check_person_output.sh outputs/person_detect.mp4 outputs/results.jsonl
```

### 17.2 JSONL 为空

可能原因：

- pipeline 没有真正跑完。
- probe 没挂到正确位置。
- `pyds` 未正确安装。
- `nvinfer` 未产生 object meta。

检查：

```bash
scripts/check_env.sh
grep -F '"detections": [{' outputs/results.jsonl | head
```

### 17.3 detection 全是 `unknown`、bbox 全是 `0`

这通常是 DeepStream metadata cast 错误。当前项目已按上下文分别 cast：

- `NvDsFrameMeta`
- `NvDsObjectMeta`

如果再次出现，优先检查是否改动过 `src/app/infrastructure/pipeline/builder.py` 中 metadata 解析相关逻辑。

### 17.4 confidence 出现 `40`、`67` 这种异常值

YOLO 置信度正常应在 `0..1`。出现大于 `1` 的 confidence，通常说明 ONNX/engine 与 YOLO parser 输出格式不匹配。

解决方式：

1. 使用 DeepStream-Yolo 的 `utils/export_yoloV8.py` 重新导出 ONNX。
2. 用新 ONNX 重新生成 TensorRT engine。
3. 确认 `infer_primary_yolo_minimal.txt` 中：

```text
parse-bbox-func-name=NvDsInferParseYolo
output-blob-names=output0
maintain-aspect-ratio=1
symmetric-padding=1
```

### 17.5 输出视频比例不符合预期

输出尺寸由运行脚本环境变量控制：

```bash
OUTPUT_WIDTH=1280 OUTPUT_HEIGHT=720 scripts/run_person_detect.sh input.mp4
```

或者修改：

```text
configs/app/app_minimal.yaml
```

中的：

```yaml
inference_width: 1280
inference_height: 720
```

注意：这里控制的是 `nvstreammux` 输出尺寸和最终编码尺寸。`nvinfer` 仍通过：

```text
infer-dims=3;640;640
```

执行 YOLO 输入尺寸推理。

### 17.6 `GStreamer element primary-infer does not support property custom-lib-path`

这是 Python 尝试把 `custom-lib-path` 当作 GStreamer element property 设置时产生的 warning。当前实际 parser 路径由 `nvinfer` 配置文件提供：

```text
custom-lib-path=...
```

只要日志里出现：

```text
Load new model: configs/deepstream/infer_primary_yolo_minimal.txt sucessfully
```

并且 JSONL 有正常检测结果，这个 warning 不阻塞当前流程。

### 17.7 `libtritonserver.so` 或 `librivermax.so.0` warning

当前最小流程不使用 Triton Inference Server 或 Rivermax UDP。只要 `nvinfer`、`nvstreammux`、`nvdsosd`、`nvv4l2decoder`、`nvv4l2h264enc` 正常，这些 warning 可暂时忽略。

## 18. 重新生成 YOLOv8s engine

如果替换了 ONNX，需要重新生成 engine：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master
source scripts/env.sh

rm -f models/yolov8s.engine

/usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolov8s.onnx \
  --saveEngine=models/yolov8s.engine \
  --fp16
```

观察进程：

```bash
ps -ef | grep trtexec | grep -v grep
```

完成后确认：

```bash
ls -lh models/yolov8s.engine
```

## 19. 使用 DeepStream-Yolo 重新导出 ONNX

如果模型输出异常，推荐使用 DeepStream-Yolo 的 YOLOv8 导出脚本。

准备导出目录：

```bash
cd /home/nvidia/Desktop/YOLO
mkdir -p export_yolov8_ds
cd export_yolov8_ds
cp /home/nvidia/Desktop/YOLO/DeepStream-Yolo-master/utils/export_yoloV8.py .
```

创建虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
```

如果系统缺少 venv：

```bash
sudo apt update
sudo apt install -y python3.10-venv
```

安装依赖：

```bash
pip install --upgrade pip
pip install ultralytics onnx onnxslim onnxruntime onnxscript
```

导出：

```bash
python3 export_yoloV8.py -w yolov8s.pt --dynamic --simplify
```

替换项目模型：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master

cp /home/nvidia/Desktop/YOLO/export_yolov8_ds/yolov8s.onnx models/yolov8s.onnx
cp /home/nvidia/Desktop/YOLO/export_yolov8_ds/labels.txt models/labels.txt
```

然后重新生成 engine。

## 20. 项目完成度与后续路线

当前项目已经进入 **离线批量验收版本基本完成** 阶段。按当前目标“本地 8 个 MP4 输入、本地 MP4 输出、person JSONL 输出、批量统计和 UI 查看结果”来评估，完成度约为 **75% 到 80%**。

已经完成的核心闭环：

```text
8 路本地 MP4 输入
-> 8 路并行 DeepStream YOLOv8 person analytics
-> 每路输出 MP4 / overlay MP4 / JSONL / analytics_summary.json / run.log
-> batch_summary.json / batch_quality.json / CSV / HTML report
-> 本地 UI 展示验收结果、质量状态、输出视频和报告入口
```

后续任务建议按照下面顺序推进。

### 20.1 性能基准与瓶颈定位

目标：

```text
确认当前 8 路并行的真实性能上限，并判断瓶颈来自哪里。
```

原因：

当前 `BATCH_JOBS=8` 已经可以跑通，但 4 路和 8 路耗时接近，说明系统可能已经遇到瓶颈。瓶颈可能来自硬件解码、TensorRT 推理、OSD、硬件编码、磁盘写入，或者 8 个独立进程之间的资源争用。

建议方案：

1. 固定同一批 8 个视频，分别跑：

```bash
BATCH_JOBS=1 scripts/run_person_analytics_batch.sh /home/nvidia/Desktop/YOLO/video outputs/bench_jobs_1
BATCH_JOBS=4 scripts/run_person_analytics_batch.sh /home/nvidia/Desktop/YOLO/video outputs/bench_jobs_4
BATCH_JOBS=8 scripts/run_person_analytics_batch.sh /home/nvidia/Desktop/YOLO/video outputs/bench_jobs_8
```

2. 对比每次输出中的：

- `total_duration_seconds`
- `processing_fps`
- 每路 `duration_seconds`
- 每路 `processing_fps`
- `batch_quality.json`

3. 后续可以新增一个 benchmark 脚本自动汇总：

```text
scripts/benchmark_batch_jobs.sh
outputs/benchmarks/benchmark_summary.json
outputs/benchmarks/benchmark_report.html
```

验收标准：

- 能明确回答 `BATCH_JOBS=1/4/8` 哪个更适合作为默认运行方式。
- 能判断 8 路并行是否真的带来收益。
- 能根据处理 FPS 区分“源视频帧率低”和“系统处理慢”。

### 20.2 多路并行架构升级

目标：

```text
从 8 个独立进程并行，升级到更接近生产的单 DeepStream pipeline 多路输入。
```

当前方案：

```text
8 个视频 -> 8 个 python/deepstream 进程 -> 8 份独立输出
```

优点是实现简单、隔离性强、容易验收；缺点是多个进程会重复加载模型、重复占用 TensorRT/GPU/编码资源，长期不适合作为真正实时多路架构。

推荐生产方向：

```text
8 个 source bin
-> nvstreammux batch-size=8
-> nvinfer batch 推理
-> tracker / analytics / osd
-> 分路输出或合成预览
```

建议方案：

1. 保留当前 8 进程批处理作为离线验收和回归测试工具。
2. 先使用 `scripts/run_multifile_inproc.sh` 验证本地 8 个 MP4 的单进程多路输入。
3. 跑通后，对比 `outputs/multifile_inproc/results.jsonl` 和当前 8 进程批处理结果。
4. 再选择视频输出方案：`nvmultistreamtiler` 合成一路预览，或 `nvstreamdemux` 分路输出。
5. 最后再把 RTSP/RTMP 摄像头接入放到真实流阶段。

验收标准：

- 单进程能同时接入 8 个本地 MP4。
- `nvstreammux batch-size=8` 正常工作。
- 每路仍能输出 `stream_id`、`track_id`、JSONL 和统计结果。
- 总处理 FPS 相比 8 独立进程有明确对比数据。

### 20.3 质量规则定稿

目标：

```text
明确什么情况算 passed、review、failed，避免 UI 只显示结果但没有验收口径。
```

当前已有质量检查：

- 输出视频是否存在。
- overlay 视频是否存在。
- JSONL 是否存在。
- analytics summary 是否存在。
- 帧是否连续。
- FPS 是否异常。
- 是否有失败原因。

建议继续定稿的规则：

- JSONL 为空：通常应为 `failed`。
- 没有 person：根据业务场景决定是 `passed` 还是 `review`。
- FPS 低：建议先设为 `review`，除非低到影响交付。
- 帧不连续：建议设为 `review` 或 `failed`，取决于断帧比例。
- 输出视频损坏：必须 `failed`。
- tracker ID 大量跳变：建议新增 `review` 规则。

验收标准：

- 每个 `review/failed` 都有明确中文原因。
- UI 和 `batch_quality.json` 的结论一致。
- 项目交付时可以解释“为什么这 8 个视频通过/复核/失败”。

### 20.4 长视频稳定性测试

目标：

```text
确认系统不只是在 10 秒短视频上可用，也能处理更长视频。
```

建议方案：

1. 准备 3 类输入：

- 1 分钟视频。
- 10 分钟视频。
- 30 分钟以上视频。

2. 运行同一套验收入口：

```bash
scripts/run_acceptance_ui.sh /path/to/long_video_dir outputs/acceptance_long
```

3. 重点观察：

- 输出视频是否完整。
- JSONL 是否持续写入。
- tracker ID 是否稳定。
- 内存是否持续增长。
- `processing_fps` 是否随时间下降。

验收标准：

- 长视频输出文件可播放。
- `batch_quality.json` 不出现异常失败。
- 处理过程无明显内存泄漏或进程卡死。

### 20.5 RTSP/RTMP 摄像头接入

目标：

```text
把当前离线 MP4 能力迁移到真实摄像头输入。
```

当前决定：

RTSP/RTMP 先保留代码方向，但不作为当前阶段优先实现。等离线批量、多路性能和质量规则稳定后再做。

如果暂时跳过长视频稳定性测试，那么下一个推荐节点不是直接接真实摄像头，而是先做：

```text
本地 MP4 -> 本地 RTSP 模拟器 -> 单进程 DeepStream RTSP 拉流
```

这个节点的价值是：在没有真实摄像头的情况下，提前验证 live-source、网络拉流、动态 pad、断流恢复、时间戳和实时帧率控制。

建议方案：

1. 如果没有真实摄像头，先用 `scripts/serve_mp4_as_rtsp.py` 把本地 MP4 暴露成 `rtsp://127.0.0.1:8554/stream1..8`。
2. 用 `scripts/run_rtsp_inproc.sh` 验证单进程 8 路 RTSP 拉流。
3. 再接 1 路真实 RTSP 摄像头。
4. 处理断流重连、网络抖动、延迟和时间戳。
5. 再扩展到多路真实 RTSP。
6. 最后考虑是否需要 RTMP/HLS/WebRTC 预览输出。

验收标准：

- 摄像头断开后能恢复或明确报错。
- UI 能显示在线、断流、重连、失败状态。
- JSONL 中时间戳能区分处理时间和源视频时间。

### 20.5.1 从多线程解码推理 demo 吸收的优化策略

项目中已有 `demo_multhread_decode_infer_mulmodel`，它是 RKNN/MPP/RGA 生态的 C++ demo，不建议直接复制到当前 DeepStream 项目中。当前项目应吸收它的工程策略，并按 DeepStream/Python 架构重新实现。

#### 1. 限长队列和丢旧帧

demo 中推流队列超过阈值后会丢弃最旧帧，避免输出端阻塞导致内存持续增长。

当前项目的适配方式：

- 不放进 DeepStream 主推理链路，主链路仍交给 `nvstreammux`、`queue` 和 DeepStream 调度。
- 后续用于 UI 实时预览队列、RTMP/RTSP 输出队列或 WebSocket 推送队列。
- 队列满时丢弃最旧帧，并记录 `frames_dropped`。

建议落点：

```text
src/app/infrastructure/web/
src/app/infrastructure/streaming/
```

验收标准：

- 输出端卡顿时，主推理链路不被阻塞。
- UI 或质量报告能看到丢帧计数。

#### 2. 超时保活最后一帧

demo 推流线程在没有新帧时会复用最后一帧，保持连接活跃。

当前项目的适配方式：

- 用于后续 RTMP/RTSP/HLS/WebRTC 输出。
- 当一段时间没有新帧时，复用上一帧，或输出黑帧加状态文字。
- 不影响 JSONL 结果，只影响视频预览输出。

验收标准：

- 短时间无新帧时，推流连接不断。
- UI 显示“复用上一帧”或“源暂时无帧”的状态。

#### 3. 本地 MP4 按源 FPS 限速

demo 对本地文件按源视频 FPS 节流，避免本地文件模拟实时流时倍速播放。

当前项目的适配方式：

- 放到 `scripts/serve_mp4_as_rtsp.py` 和 `scripts/serve_mp4_as_rtsp_loop.py`。
- FFmpeg 方案优先使用 `-re`。
- GStreamer 方案使用 clock/sync 机制，例如 `identity sync=true` 或等价方式。
- 记录源视频 FPS、推流 FPS 和 loop 次数。

验收标准：

- MP4 模拟 RTSP 时接近真实摄像头帧率。
- 单进程 RTSP 拉流看到的是 live-source 行为，而不是离线极速读取。

#### 4. 硬件路径失败后的 fallback

demo 中 RGA 失败后回退 OpenCV，并限频打印错误。

当前项目的适配方式：

- DeepStream 主链路优先使用硬件解码、推理、OSD、编码。
- 非主链路工具允许 fallback，例如 overlay 生成失败时保留原始 MP4，或生成 JSONL-only 结果。
- 所有 fallback 都写入 `run_metadata.json`、`batch_quality.json` 或 `multifile_quality.json`。

当前已完成：

- probe 热路径异常已经增加限频日志。
- GStreamer/pyds 导入失败会记录 warning。
- 单 pipeline 失败也会生成 summary/quality。

#### 5. 流读取失败后的重连策略

demo 对本地文件 EOF 立即 reopen，对网络流失败延迟重连。

当前项目的适配方式：

- 本地 MP4 模拟 RTSP：播完立即 loop。
- RTSP 网络流：断流后 5 到 10 秒重试。
- 连续失败超过阈值后标记该路 failed，但其他路继续运行或至少生成清晰质量报告。

建议新增状态文件：

```text
source_status.json
```

建议字段：

```json
{
  "stream_id": "stream-0",
  "uri": "rtsp://127.0.0.1:8554/stream1",
  "status": "online|reconnecting|failed|eos",
  "last_frame_at": "...",
  "last_error": "",
  "restart_count": 0,
  "consecutive_failures": 0
}
```

#### 6. 资源释放集中化

demo 的 decoder、encoder、streaming manager 都有 stop/release 思路。

当前项目的适配方式：

- 继续强化 `PipelineManager.stop()`、`DashboardServer.stop()`、`JsonWriter.close()`、后续 streaming server stop。
- 所有后台线程必须有停止信号、唤醒机制和 join。

当前已完成：

- bus 线程 stop 时 join。
- dashboard HTTP 线程 stop 时 join。
- frame result handler 异常不会打穿 probe 回调。

#### 7. 多模型融合

demo 的 `DetectionFusionManager` 使用 IoU、IoM、加权融合做多模型结果合并。

当前项目的适配方式：

- 先不放进当前 person-only 落地版。
- 后续扩展安全帽、打电话、疲劳、摔倒等模型时再新增 fusion 层。
- 建议新增：

```text
src/app/domain/fusion.py
src/app/application/fusion_service.py
configs/analytics/fusion.yaml
```

融合策略：

- 同类目标用 IoU/NMS 去重。
- 人体框与安全帽等嵌套小框用 IoM。
- 多模型框用 weighted box fusion。
- 输出 `fused_detections` 到 JSONL 和 UI。

### 20.5.2 跳过长跑验收后的下一节点

如果当前明确跳过“长视频稳定性测试”，下一节点建议做：

```text
MP4 模拟 RTSP 的真实时序和循环重连
```

目标：

- 本地 8 个 MP4 以接近真实摄像头的方式推成 8 路 RTSP。
- 每路 MP4 按源 FPS 或指定 FPS 推流，不倍速。
- 每路播完自动 loop。
- 推流进程异常退出后能重启或明确写出失败状态。
- 生成 `source_status.json`，供 UI 或验收脚本读取。

建议先实现脚本层能力：

```text
scripts/serve_mp4_as_rtsp_loop.py
scripts/run_rtsp_inproc.sh
```

验收命令目标：

```bash
python3 scripts/serve_mp4_as_rtsp_loop.py \
  --input-dir /home/nvidia/Desktop/YOLO/video \
  --host 127.0.0.1 \
  --port 8554 \
  --loop \
  --realtime

scripts/run_rtsp_inproc.sh \
  rtsp://127.0.0.1:8554 \
  outputs/rtsp_inproc
```

验收标准：

- 能看到 8 路 RTSP URL。
- DeepStream 单进程能从 RTSP URL 拉流。
- `outputs/rtsp_inproc/results.jsonl` 中出现 `stream-0..stream-7`。
- 关闭某一路模拟器后，状态能变为 reconnecting 或 failed。
- 恢复该路后，状态能重新变为 online。

### 20.6 服务化与部署收口

目标：

```text
让项目从“终端手动运行”变成“设备上稳定运行”。
```

建议方案：

- systemd 服务。
- 开机自启动。
- 日志轮转。
- 输出目录自动清理。
- 异常退出自动恢复。
- 固定配置文件和模型路径。
- 固定 Jetson 环境依赖版本。

验收标准：

- 重启 Jetson 后服务能自动启动。
- UI 能访问最新结果。
- 错误日志可追踪。
- 输出文件不会无限增长占满磁盘。

### 20.7 最终交付文档

目标：

```text
让别人拿到 Jetson 后，能按照文档独立运行、验收和排查。
```

最终文档至少需要包含：

- 一条命令运行验收。
- UI 页面说明。
- 输入视频目录规范。
- 输出文件说明。
- 模型和 engine 重新生成方法。
- 质量状态解释。
- 常见错误排查。
- Jetson 环境依赖版本。
- 性能基准结果。

验收标准：

- 新用户不依赖开发者口头说明，也能跑通离线 8 路验收。
- 遇到常见错误能按文档定位到环境、模型、parser、视频编码或路径问题。
