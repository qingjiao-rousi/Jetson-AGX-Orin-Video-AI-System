# 主模型 FP16/INT8 离线检测对齐

系统 benchmark 的检测数、事件数和跟踪数会同时受到帧丢弃、tracker、任务路由和
专用 worker 的影响，不能用于判定主模型 INT8 的检测差异。本实验将这些变量排除：
同一张输入帧依次经过 FP16、INT8 主 YOLO engine，比较相同后处理后的检测框。

## 固定口径

- 输入：显式指定的本地图像目录/文件或按固定 stride 抽取的视频帧；执行后输出会记录
  输入路径、engine SHA256 与逐帧结果。
- 预处理：BGR -> RGB、letterbox 到 `640x640`、CHW、float32、`[0, 1]`。
- 后处理：置信度阈值 `0.25`、class-aware NMS IoU `0.45`。
- 默认仅保留 class `0`（person），与当前主 DeepStream 配置的 `filter-out-class-ids=1..79`
  一致。若需检验所有 COCO 类，显式传入 `--class-ids 0,1,...,79`。
- 匹配：同类别、一对一、IoU >= `0.50`。不经过 DeepStream、tracker、路由、队列、
  专用模型或输出 sink。

因此它衡量的是当前 engine 和统一预/后处理下的**部署检测输出一致性**，不是带标注的
mAP、Precision 或 Recall。

## 执行

先用与 INT8 校准和实际场景都相关的一批固定帧；推荐将其放在本地忽略目录
`calibration/alignment/`，不要提交视频帧或结果文件。

```bash
cd <repository-root>
source scripts/deploy/env.sh

python3 scripts/evaluation/align_primary_detector_outputs.py \
  --input calibration/alignment \
  --fp16-engine models/fp16/yolov8s.engine \
  --int8-engine models/int8/yolov8s_int8.engine \
  --max-frames 300
```

也可从固定视频抽帧；`--video-stride 30` 表示每 30 帧取一帧：

```bash
python3 scripts/evaluation/align_primary_detector_outputs.py \
  --input video/1.mp4 video/2.mp4 \
  --video-stride 30 \
  --max-frames 300
```

输出位于 `outputs/precision_alignment/<UTC时间>/`，其中 `summary.json` 是汇总，
`frame_comparisons.jsonl` 保存逐帧的 FP16/INT8 检测框、匹配和仅一侧存在的框，便于
抽样人工复核。两者都已被 Git 忽略。

## 判读和后续行动

必须同时报告：`fp16_total`、`int8_total`、`matched`、`fp16_only`、`int8_only`、
匹配 IoU 分布，以及 `INT8 - FP16` 置信度差。若 only 框主要集中在 0.25 阈值附近，
应额外记录阈值敏感性；若匹配 IoU 或差异较差，先检查是否使用同一 ONNX、输入尺寸、
预处理和 TensorRT 版本，再决定是否更换校准集、调整量化方法或保留 FP16。

## 解码器修正说明

2026-08-11 的初版对齐结果无效，已撤回：初版错误地将主模型输出按 PPE 的
`[4 + classes, anchors]` 格式解析。实际 engine 输出由
`export_yolov8_ds/export_yoloV8.py`（派生自上游 DeepStream-Yolo）的输出契约，为
`[x1, y1, x2, y2, score, class_id]`，shape 为 `[batch, 8400, 6]`，且未做 NMS。
FP16 与 INT8 engine 已在 Jetson 上实际确认这一 shape。

对齐工具现已使用主模型专用解码、统一 class-aware NMS 和同一 letterbox 逆变换。必须
重新执行本文档中的命令后，才能填写 FP16/INT8 一致性结论。此前的 `566/422` 检测数、
匹配率、IoU 和置信度差均不应引用。
