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
scripts/smoke_test_person_detect.sh
```

一键烟测脚本。

```text
scripts/smoke_test_person_tracker.sh
```

一键 tracker 烟测脚本，检查 JSONL 中是否存在稳定 `track_id`。

```text
scripts/summarize_person_tracks.py
```

离线统计脚本，读取 tracker JSONL 并生成 `summary.json`。

```text
scripts/smoke_test_person_counting.sh
```

一键计数烟测脚本，串联检测、跟踪、统计。

```text
scripts/summarize_person_roi.py
```

离线 ROI 统计脚本，使用 `track_id + bbox 中心点` 判断人员是否在区域内。

```text
scripts/smoke_test_person_roi.sh
```

一键 ROI 烟测脚本，串联检测、跟踪、ROI 统计。

```text
scripts/summarize_person_line.py
```

离线越线计数脚本，使用 `track_id + bbox 中心点轨迹` 判断是否穿过一条线。

```text
scripts/smoke_test_person_line.sh
```

一键越线计数烟测脚本。

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
scripts/smoke_test_person_analytics.sh
```

一键 analytics 烟测脚本，串联检测、跟踪、统一统计和可视化叠加。

```text
scripts/summarize_person_timeline.py
```

离线时间轴与同步基准脚本，检查 `stream_id/frame_id/timestamp` 的连续性、单调性和估算 FPS。

```text
scripts/smoke_test_person_timeline.sh
```

一键 timeline 烟测脚本。

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

## 6. 一键烟测

推荐每次修改代码后运行烟测：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master

scripts/smoke_test_person_detect.sh
```

默认测试视频：

```text
/home/nvidia/Desktop/YOLO/video/1.mp4
```

指定测试视频：

```bash
scripts/smoke_test_person_detect.sh /home/nvidia/Desktop/YOLO/video/2.mp4
```

烟测输出默认写到：

```text
outputs/smoke/person_detect.mp4
outputs/smoke/results.jsonl
```

烟测会自动执行：

1. 运行 person 检测。
2. 检查 MP4 是否存在且可被 GStreamer 识别。
3. 检查分辨率是否符合预期。
4. 检查 JSONL 是否非空。
5. 检查 JSONL 是否只包含 `person`。
6. 检查 bbox 是否非空。
7. 检查 confidence 是否在 `0..1` 合理范围。

看到下面结果说明最小闭环正常：

```text
Failures: 0
Smoke test passed.
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

运行 tracker 烟测：

```bash
scripts/smoke_test_person_tracker.sh
```

指定视频：

```bash
scripts/smoke_test_person_tracker.sh /home/nvidia/Desktop/YOLO/video/2.mp4
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

一键计数烟测：

```bash
scripts/smoke_test_person_counting.sh
```

指定视频：

```bash
scripts/smoke_test_person_counting.sh /home/nvidia/Desktop/YOLO/video/2.mp4
```

默认输出：

```text
outputs/smoke/person_counting.mp4
outputs/smoke/counting_results.jsonl
outputs/smoke/summary.json
```

验收通过时会看到：

```text
[OK] Counting summary is valid
Counting smoke test passed.
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

一键 ROI 烟测：

```bash
scripts/smoke_test_person_roi.sh
```

指定 ROI：

```bash
ROI=0,0,640,720 ROI_ID=left-half scripts/smoke_test_person_roi.sh
```

默认输出：

```text
outputs/smoke/person_roi.mp4
outputs/smoke/roi_results.jsonl
outputs/smoke/roi_summary.json
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

一键越线烟测：

```bash
scripts/smoke_test_person_line.sh
```

指定计数线：

```bash
LINE=640,0,640,720 LINE_ID=middle-vertical scripts/smoke_test_person_line.sh
```

默认输出：

```text
outputs/smoke/person_line.mp4
outputs/smoke/line_results.jsonl
outputs/smoke/line_summary.json
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

可以通过 `MIN_SIDE_DISTANCE` 或 `--min-side-distance` 忽略贴近线附近的小抖动：

```bash
MIN_SIDE_DISTANCE=8 scripts/smoke_test_person_line.sh
```

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

一键 analytics 烟测：

```bash
scripts/smoke_test_person_analytics.sh
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

一键 smoke test：

```bash
scripts/smoke_test_person_timeline.sh
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

如果 `/home/nvidia/Desktop/YOLO/video` 中有多个 MP4，可以使用批量入口顺序处理。当前批量模式是 **离线多文件批处理**：

```text
多个本地 MP4 -> 逐个调用单视频 analytics -> 每个视频独立输出 -> 汇总 batch_summary.json
```

它不是实时多路摄像头同时拉流，也不是 DeepStream 单 pipeline 多 source 并发处理。这样做的好处是先把八个离线视频稳定跑通，便于验收模型、tracker、ROI、越线、时间轴和输出文件。

运行八个视频：

```bash
cd /home/nvidia/Desktop/YOLO/Jetson-agx-orin-32g-YOLO-master

scripts/run_person_analytics_batch.sh \
  /home/nvidia/Desktop/YOLO/video \
  outputs/batch
```

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

## 20. 下一步建议

当前 tracker、离线去重统计、ROI 统计、越线计数、统一 analytics summary、可视化叠加、离线时间轴和批量视频处理都已具备后，建议下一阶段先做 **批量结果报表与质量对比**。

目标：

```text
batch videos -> batch_summary.json -> CSV/HTML report -> 人工复核 -> 参数调优
```

原因是：RTSP/RTMP 摄像头输入已经明确放到最后再做，当前更值得先把八个离线视频的结果变成可比较、可复核、可交付的报表。

建议后续先补：

- `batch_summary.csv`：每个视频一行，包含去重人数、ROI 人数、越线 in/out、帧连续性、估算 FPS、运行状态。
- `batch_report.html`：可打开的离线报告，包含每个视频的输出链接、关键统计和失败原因。
- 批量质量检查：标记 JSONL 为空、无 track、无 person、时间轴不连续、视频输出不可解析等异常。
- 参数复核：基于八个视频结果调整 `CONFIDENCE_THRESHOLD`、ROI、line 和输出尺寸。

再往后才适合进入：

1. 长视频稳定性测试。
2. 性能记录与资源占用统计。
3. RTSP/RTMP 摄像头拉流。
4. 多路实时输入。
5. 服务化与部署收口。
