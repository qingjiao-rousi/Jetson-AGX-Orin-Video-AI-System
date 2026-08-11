# COCO val2017 主模型 FP16/INT8 标注评测

本评测补足离线输出对齐的边界：对齐只说明 INT8 相对 FP16 的变化；COCO val2017
带有 ground truth，能报告两份 engine 各自的标注质量指标和两者差值。

## 口径

- 数据集：COCO `val2017` 与 `annotations/instances_val2017.json`；完整标准评测必须使用
  全部 5,000 张图。`--max-images` 仅用于冒烟检查，不得称为完整 COCO 结果。
- 类别：运行时严格验证 `models/labels.txt` 是否能映射到 annotation 中的 COCO 类别 ID；
  不接受仅凭类别索引直接写入预测结果。
- 预处理：BGR -> RGB、letterbox `640x640`、CHW、float32、`[0,1]`。
- 后处理：主 engine 的实际输出为 `[x1, y1, x2, y2, score, class_id]`，先恢复
  letterbox 坐标，再做 class-aware NMS IoU `0.45` 和全局 top-k `300`，与项目主
  DeepStream 配置保持一致。COCO AP 使用低分数地板 `0.001`，避免固定部署阈值截断 PR
  曲线。
- 标准质量：COCOeval `bbox` 的 AP、AP50、AP75 及小/中/大目标 AP，分别汇总全类别和
  `person` 类。
- 业务工作点：固定 FP16 阈值 `0.25`，扫描 INT8 `0.25/0.20/0.15/0.10`，以非 crowd
  person 标注和 IoU=0.50 做置信度降序的一对一匹配，输出 Precision、Recall、F1。
  它是部署工作点指标，不替代 COCOeval 对 crowd/ignore 的完整协议处理。

## 获取数据与依赖

在 Jetson 的本地忽略目录保存数据；COCO 图像、标注和预测 JSON 均不提交 Git。下载大小
约为 1GB 图像加数百 MB 标注，请确认磁盘空间与网络后手动执行：

```bash
cd <repository-root>
mkdir -p datasets/coco
cd datasets/coco
wget https://images.cocodataset.org/zips/val2017.zip
wget https://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip 'annotations/instances_val2017.json'

cd <repository-root>
python3 -m pip install -r requirements-eval.txt
```

## 冒烟与完整评测

先确认格式、label 映射和 engine 可用；该 50 张子集不能作为公开结果：

```bash
cd <repository-root>
source scripts/env.sh

python3 scripts/evaluate_primary_coco.py \
  --images-dir datasets/coco/val2017 \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --max-images 50
```

确认无误后，删除 `--max-images`，执行完整 5,000 张 COCO val2017：

```bash
python3 scripts/evaluate_primary_coco.py \
  --images-dir datasets/coco/val2017 \
  --annotations datasets/coco/annotations/instances_val2017.json
```

结果位于 `outputs/coco_eval/<UTC时间>/`，包含 FP16/INT8 COCO prediction JSON 与
`summary.json`。完整评测后应记录 engine SHA256、软件版本和评测命令，并在报告中并列：

- FP16/INT8 全类别 AP、AP50、AP75；
- FP16/INT8 person AP、AP50、AP75；
- 固定 FP16=0.25 与每个 INT8 阈值下的 person Precision、Recall、F1；
- INT8 相对 FP16 的绝对差值（百分点），不只报告相对百分比。

如果 INT8 AP 接近 FP16、但低阈值使 person Recall 恢复且 Precision 可接受，则可将
INT8 阈值作为部署参数单独调优。若 INT8 的 person AP 和低阈值 Recall 仍显著较低，则
优先检查校准集和重建 INT8 engine；不要用阈值掩盖质量损失。

## 2026-08-11 完整评测结果

结果来源：`outputs/coco_eval/20260811T080129Z/summary.json`。在 Jetson 上完成完整
COCO val2017（5,000 张）评测；FP16/INT8 engine SHA256 分别为
`f69e84e8bee99ccd2627a7c8331552df4e627ec23f499e13315378bc1c855079` 与
`e036e71f8b3717c5abe297308fb9437a5ab2baa91dd79ad117cfe453b782ed70`。

| 指标 | FP16 | INT8 | INT8 - FP16 |
| --- | ---: | ---: | ---: |
| 全类别 AP | 0.4340 | 0.3659 | -6.81 个百分点 |
| 全类别 AP50 | 0.6099 | 0.5295 | -8.03 个百分点 |
| 全类别 AP75 | 0.4617 | 0.3936 | -6.81 个百分点 |
| person AP | 0.5704 | 0.5258 | -4.46 个百分点 |
| person AP50 | 0.8181 | 0.7823 | -3.58 个百分点 |
| person AP75 | 0.6062 | 0.5568 | -4.94 个百分点 |

固定 FP16 部署阈值 0.25 时，person 的 FP16 工作点为 Precision `0.7892`、Recall
`0.7286`、F1 `0.7577`。INT8 阈值扫描如下：

| INT8 阈值 | Precision | Recall | F1 | 相对 FP16 Recall / F1 差值 |
| ---: | ---: | ---: | --- |
| 0.25 | 0.8714 | 0.6050 | 0.7142 | -12.36 / -4.35 个百分点 |
| 0.20 | 0.8417 | 0.6364 | 0.7248 | -9.22 / -3.29 个百分点 |
| 0.15 | 0.7977 | 0.6717 | 0.7293 | -5.69 / -2.84 个百分点 |
| 0.10 | 0.7249 | 0.7109 | 0.7178 | -1.77 / -3.99 个百分点 |

结论：当前 INT8 engine 的 `person AP` 损失 4.46 个百分点，超过本项目设定的 3 个
百分点候选门槛；阈值下调能恢复 Recall，但不能同时恢复 FP16 的 Recall 与 F1。当前
质量基线和默认部署应保持 FP16。INT8 仍可作为吞吐/功耗研究对象，但在重建并评测新的
校准 engine 前，不应表述为 FP16 的低风险等价替代品。

### COCO train504 校准候选结果

使用下文所述的独立 COCO train2017 504 张校准集重新构建的候选，结果来源为
`outputs/coco_eval/20260811T090921Z/summary.json`。它相对旧 INT8 有明显改善：全类别
AP 从 `0.3659` 提升到 `0.3984`，person AP 从 `0.5258` 提升到 `0.5503`。

| person 指标 | FP16 | 旧 INT8 | train504 INT8 | 候选相对 FP16 |
| --- | ---: | ---: | ---: | ---: |
| AP | 0.5704 | 0.5258 | 0.5503 | -2.01 个百分点 |
| AP50 | 0.8181 | 0.7823 | 0.8073 | -1.07 个百分点 |
| AP75 | 0.6062 | 0.5568 | 0.5882 | -1.81 个百分点 |

| 工作点 | Precision | Recall | F1 | 相对 FP16 Recall / F1 差值 |
| --- | ---: | ---: | ---: | --- |
| FP16 @ 0.25 | 0.7892 | 0.7286 | 0.7577 | 基线 |
| 候选 INT8 @ 0.25 | 0.8715 | 0.6338 | 0.7339 | -9.48 / -2.38 个百分点 |
| 候选 INT8 @ 0.20 | 0.8373 | 0.6722 | 0.7457 | -5.64 / -1.20 个百分点 |
| 候选 INT8 @ 0.15 | 0.7859 | 0.7116 | 0.7469 | -1.70 / -1.08 个百分点 |
| 候选 INT8 @ 0.10 | 0.7117 | 0.7482 | 0.7295 | +1.96 / -2.82 个百分点 |

初步决策：候选 INT8 在 `0.15` 是当前平衡工作点，满足本项目的候选门槛（person AP
损失约 2 个百分点、Recall/F1 损失均低于 3 个百分点）。`0.10` 是 Recall 优先工作点，
但会更显著降低 Precision 并增加下游任务量。候选 engine 已获准进入 8 路系统复测；在
完成系统复测和真实业务标注帧验证前，默认部署仍保持 FP16。

## 独立 COCO train2017 校准候选

这不是“用 500 张图训练 INT8 模型”。TensorRT PTQ 校准仅使用无标注输入生成量化 scale，
不更新 YOLO 权重。由于 `val2017` 已用于本项目的完整质量评测，**禁止**再从其中抽图做
校准；应使用独立的 `train2017`。

当前主 engine 的校准 batch 是 8。校准器不会使用最后一个不完整 batch，因此选用 504 张
（而不是恰好 500 张）以让所有图片参与校准。该候选用于回答“COCO train 分布的校准是否
能改善 COCO val 的 INT8 质量”，不能替代后续真实厂区场景校准。

先从已下载的 annotation ZIP 解出 train 标注：

```bash
cd <repository-root>/datasets/coco
unzip annotations_trainval2017.zip 'annotations/instances_train2017.json'

cd <repository-root>
source scripts/env.sh
python3 scripts/prepare_coco_train_calibration.py \
  --annotations datasets/coco/annotations/instances_train2017.json
```

脚本会在本地忽略目录 `calibration/coco_train504/` 写入固定随机种子（`20260811`）的
`manifest.json` 和 `download_urls.txt`。它不会下载文件。下载 504 张 selected train 图片：

```bash
mkdir -p calibration/coco_train504/images
wget --no-check-certificate --continue \
  -i calibration/coco_train504/download_urls.txt \
  -P calibration/coco_train504/images

find calibration/coco_train504/images -maxdepth 1 -type f -name '*.jpg' | wc -l
```

最后一条应为 `504`。`--no-check-certificate` 仅是当前网络证书异常下、下载公开 COCO
文件的临时做法；可修复 TLS 后应移除该选项。

确认导出的 ONNX 与当前 engine 都采用 `[batch,8400,6]` 输出后，以新文件名构建候选，
绝不覆盖当前 baseline INT8 engine：

```bash
python3 scripts/build_yolov8s_int8.py \
  --onnx export_yolov8_ds/yolov8s.onnx \
  --images calibration/coco_train504/images \
  --batch-size 8 \
  --min-batch-size 1 \
  --opt-batch-size 8 \
  --max-batch-size 8 \
  --cache models/int8/yolov8s_coco_train504_calibration.cache \
  --engine models/int8/yolov8s_coco_train504.engine
```

使用同一完整 val2017 重新评测候选：

```bash
python3 scripts/evaluate_primary_coco.py \
  --images-dir datasets/coco/val2017 \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --int8-engine models/int8/yolov8s_coco_train504.engine
```

只有候选在完整 COCO `person AP`、AP75、Recall/F1 上改善后，才进入真实厂区标注帧和
多路系统复测。COCO train 候选、当前 domain 校准候选和 FP16 都应保留不同文件名并记录
engine SHA256，以便可归因比较。
