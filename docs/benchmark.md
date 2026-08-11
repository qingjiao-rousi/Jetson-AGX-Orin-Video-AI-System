# Jetson 性能实验规范

本项目将性能实验分为两条链路：`fake` 输出用于定位推理、路由和 metadata 处理的压力；`file` 输出保留硬件编码与文件写入，用于验证完整交付链路。二者必须分别比较，不能把 `fake` 的 FPS 当成完整视频输出性能。

## 测量范围

`runtime_metrics.jsonl` 的延迟定义固定为 `primary_infer_sink_to_json_write_ms`：从 `primary-infer:sink` 的帧到达时刻开始，到对应 `FrameResult` 已成功写入 `results.jsonl` 为止。

- `pipeline`：主推理前探针到 Python 结果回调。
- `json_writer`：结果回调到异步 JSON writer 成功落盘。
- `end_to_end`：以上两段的总和；汇总 `P50/P95`。

它不表示相机曝光到屏幕显示、RTSP 网络传输或编码器输出延迟。对本地 MP4，文件 PTS 也不应被当作真实世界采集时刻。

## 实验矩阵

固定矩阵为 `1/4/8` 路 x `主 YOLO FP16/INT8` x `fake/file` x `3` 次，共 36 次。安全帽、姿态、烟火、车牌检测与 OCR 在两组中均保持 FP16。每一组必须使用同一批输入视频、模型阈值、任务路由、`nvpmodel`、散热条件和软件版本。

TensorRT plan 不跨小版本前向兼容。当前运行时为 TensorRT 10.7 时，所有参与 FP16 对照的 engine 也必须由 10.7 构建。升级后先在真实 Jetson 终端运行：

```bash
scripts/build_fp16_engines_trt107.sh
```

脚本采用临时 engine 文件，只有构建成功才替换现有文件。engine 是本机生成物，仍不提交 Git。

主 YOLO 是唯一需要支持 `1/4/8` batch 的 engine，因为它位于 `nvstreammux -> primary-infer` 主链路。安全帽、姿态、烟火、车牌检测与 OCR 由 Python worker 逐个提交 ROI，保持各自 batch-1 engine；它们不因视频路数变化而重建。重建脚本默认会跳过已匹配当前 TensorRT 版本的 engine：

```bash
# 只在主 YOLO 更换或其 batch profile 变化时使用。
scripts/build_fp16_engines_trt107.sh --primary-only

# 只有专用 ONNX 或 TensorRT 版本变化时才使用。
scripts/build_fp16_engines_trt107.sh --specialists-only
```

后续若验证专用模型微批，仅将安全帽、姿态和烟火纳入范围；车牌检测与 OCR 的两阶段、可变数量链路暂不改造，也不进入该阶段的性能对比。

每次执行前记录：

| 项目 | 必须记录 |
| --- | --- |
| 环境 | Git commit、JetPack/DeepStream/TensorRT 版本、`nvpmodel -q`、`jetson_clocks` 状态 |
| 输入 | 视频文件 SHA256、路数、时长、分辨率、帧率 |
| 配置 | FP16/INT8 YAML、engine SHA256、sink、batch size、预热规则 |
| 性能 | 聚合 FPS、每路 FPS、P50/P95、GPU/RAM、GPU+SOC 功耗、温度 |
| 背压 | writer/task-buffer/frame-store/FPS controller 的累计丢弃数与 writer 错误数 |
| 有效性 | 程序退出码、EOS/错误日志、JSONL 行数、file sink 的输出视频可播放性 |

不要在同一张表中混用不同视频、不同功耗模式或不同 sink 的结果。

## 执行

先只生成配置和计划，检查后再实跑：

```bash
cd <repository-root>
python3 scripts/run_benchmark_matrix.py
```

确认本机 FP16/INT8 engine、八路 MP4 和散热状态都准备好后：

```bash
python3 scripts/run_benchmark_matrix.py --execute
```

输出位于 `outputs/benchmarks/<UTC时间>/`。每次运行保存独立的 `config.yaml`、`summary.json`（成功时）和总 `matrix_summary.json`。输出目录已被 Git 忽略。

若需要先小规模验证，可显式缩小矩阵；该结果不应写入正式 36 组表：

```bash
python3 scripts/run_benchmark_matrix.py \
  --stream-counts 1 --sinks fake --repetitions 1 --execute
```

## 2026-08-10 主模型 FP16/INT8 基线

结果来源：`outputs/benchmarks/20260810T113013Z/matrix_summary.json`。共执行 36 次，全部退出码为 0；每行是相同组合的 3 次重复平均值。

- 环境：JetPack 6.2.1 / Jetson Linux R36.4.7、TensorRT 10.7.0.23、`nvpmodel` 为 `MODE_30W`。
- 对照变量：主 YOLO 是唯一精度变量；安全帽、姿态、烟火、车牌检测和 OCR 在两组中均为 FP16、batch=1。
- 延迟口径：`primary-infer:sink` 至 `results.jsonl` 成功写入；不包含相机采集、RTSP 网络和显示延迟。
- 丢弃字段为每次运行的平均累计值，格式为 `writer/task/frame/FPS controller`；功耗为 `GPU+SoC` 电源域，并非整板功耗。

| 主模型 | 路数 | Sink | 重复 | FPS | E2E P50/P95 (ms) | GPU (%) | RAM (MB) | 功耗 (mW) | 温度 (C) | 丢弃 W/T/F/C | 结论 |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| FP16 | 1 | fake | 3 | 104.83 | 15.97 / 25.06 | 60.98 | 12287.39 | 7284.26 | 52.58 | 0 / 0 / 0 / 0 | 单路基线 |
| INT8 | 1 | fake | 3 | 105.63 | 15.98 / 26.46 | 63.16 | 12350.08 | 5748.28 | 55.95 | 0 / 0 / 0 / 0 | 吞吐接近，功耗更低 |
| FP16 | 1 | file | 3 | 104.67 | 15.75 / 26.97 | 60.64 | 12304.95 | 7472.39 | 53.88 | 0 / 0 / 0 / 0 | 单路完整输出基线 |
| INT8 | 1 | file | 3 | 104.92 | 15.88 / 30.87 | 58.89 | 12485.76 | 6107.57 | 56.15 | 0 / 0 / 0 / 0 | 吞吐接近，功耗更低 |
| FP16 | 4 | fake | 3 | 64.09 | 77.10 / 278.90 | 49.27 | 13058.58 | 6053.26 | 54.68 | 0 / 98.00 / 5502.00 / 0 | 多路队列开始积压 |
| INT8 | 4 | fake | 3 | 68.53 | 65.36 / 279.63 | 43.84 | 13134.02 | 5267.02 | 55.97 | 0 / 0.33 / 5502.00 / 0 | 吞吐提高，任务积压降低 |
| FP16 | 4 | file | 3 | 65.10 | 78.17 / 280.91 | 47.06 | 13104.89 | 6100.65 | 55.40 | 0 / 84.67 / 5502.00 / 0 | 多路完整输出基线 |
| INT8 | 4 | file | 3 | 67.45 | 69.73 / 279.08 | 34.95 | 13166.89 | 5386.22 | 55.74 | 0 / 0.33 / 5502.00 / 0 | 吞吐提高，任务积压降低 |
| FP16 | 8 | fake | 3 | 63.90 | 153.00 / 483.33 | 46.83 | 13634.70 | 5858.75 | 55.89 | 0 / 4493.33 / 14094.00 / 0 | 专用任务与帧存储压力明显 |
| INT8 | 8 | fake | 3 | 70.92 | 106.49 / 458.58 | 40.92 | 13723.72 | 5289.17 | 55.81 | 0 / 1415.00 / 14094.00 / 0 | 吞吐提高，任务积压降低 |
| FP16 | 8 | file | 3 | 65.28 | 149.56 / 476.75 | 45.39 | 13641.07 | 5904.46 | 56.29 | 0 / 4499.67 / 14094.00 / 0 | 八路完整输出基线 |
| INT8 | 8 | file | 3 | 69.03 | 110.26 / 473.63 | 37.50 | 13809.32 | 5196.70 | 55.85 | 0 / 1429.33 / 14094.00 / 0 | 吞吐提高，任务积压降低 |

## 2026-08-11 COCO train504 INT8 候选系统验证

本节不是上面的原始 36 组 baseline，而是对完成独立 COCO train2017 校准的候选 engine
进行的最终 8 路系统验证。对照固定为 FP16 主模型阈值 `0.25`，候选 INT8 主模型阈值
`0.15`；姿态、PPE、烟火、车牌检测和 OCR 均保持 FP16，PPE worker 保持 batch=1。
`fake/file` 各执行 3 次，全部退出码为 0，writer 丢弃为 0，结果写入无 unmatched。

结果来源：`outputs/benchmarks/coco_train504_int8_t015/20260811T093825Z/matrix_summary.json`。

| 主模型 | Sink | FPS | E2E P50/P95 (ms) | GPU (%) | RAM (MB) | 功耗 (mW) | 温度 (C) | Task 丢弃 | Helmet 丢弃 | FrameStore 丢弃 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 @ 0.25 | fake | 63.56 | 145.68 / 499.14 | 42.17 | 13081.70 | 5778.21 | 56.29 | 4498 | 2986 | 14094 |
| INT8 @ 0.15 | fake | 68.34 | 126.74 / 471.00 | 36.21 | 13132.98 | 5106.38 | 55.44 | 3472 | 2224 | 14094 |
| FP16 @ 0.25 | file | 63.17 | 145.73 / 498.20 | 42.29 | 13233.50 | 5870.25 | 56.94 | 4356 | 2928 | 14094 |
| INT8 @ 0.15 | file | 67.14 | 128.43 / 475.47 | 37.58 | 13214.03 | 5185.19 | 55.71 | 3498 | 2279 | 14094 |

相对同一轮 FP16 对照，候选 INT8 的 fake/file FPS 分别提高约 7.5%/6.3%，E2E P50
分别降低约 13.0%/11.9%，GPU+SoC 功耗分别降低约 11.6%/11.7%。task-buffer 总丢弃
分别降低约 22.8%/19.7%，但 FrameStore 丢弃没有变化，说明帧存储仍是独立瓶颈。

这组结果支持候选 INT8 进入真实业务帧验证和受控部署评估，但不应把它与 FP16 宣称为
完全等价：COCO person Recall 仍低约 1.7 个百分点，且较低阈值会改变下游任务触发量。
当前公开/默认配置继续保留 FP16；候选 INT8 作为已完成质量与系统验证的可选部署方案。

这些结果只证明指定系统配置下的端到端吞吐、资源和队列行为；没有标注集和逐帧输出对齐，不能据此推导 INT8 的 mAP、Precision、Recall 或精度损失。
