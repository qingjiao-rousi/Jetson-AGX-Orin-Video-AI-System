#!/usr/bin/env bash
set -euo pipefail
# ===========================================================================
# run_campus_surveillance.sh — 园区监控完整演示
#
# 一键启动: 摄像头模拟 → DeepStream 8路推理 → Web UI 验收
#
# 用法:
#   scripts/run_campus_surveillance.sh VIDEO_DIR [OUTPUT_DIR]
#
# 示例:
#   scripts/run_campus_surveillance.sh /home/nvidia/Desktop/YOLO/video
#   scripts/run_campus_surveillance.sh /home/nvidia/Desktop/YOLO/video outputs/campus_demo
# ===========================================================================

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VIDEO_DIR="${1:-/home/nvidia/Desktop/YOLO/video}"
OUTPUT_DIR="${2:-outputs/campus_surveillance}"

RTSP_PORT="${RTSP_PORT:-8554}"
SOURCE_COUNT="${SOURCE_COUNT:-8}"
MOUNT_PREFIX="${MOUNT_PREFIX:-cam}"

# ---- 清理函数 ----
cleanup() {
    echo ""
    echo "Shutting down..."
    # 停 DeepStream (通过 kill python 进程)
    if [ -n "${DS_PID:-}" ] && kill -0 "$DS_PID" 2>/dev/null; then
        kill "$DS_PID" 2>/dev/null || true
        wait "$DS_PID" 2>/dev/null || true
    fi
    # 停摄像头模拟
    bash "$ROOT_DIR/scripts/simulate_cameras.sh" --stop 2>/dev/null || true
    echo "All services stopped."
}
trap cleanup EXIT INT TERM

# ---- 参数校验 ----
if [ ! -d "$VIDEO_DIR" ]; then
    echo "ERROR: VIDEO_DIR not found: $VIDEO_DIR" >&2
    echo "Usage: $0 VIDEO_DIR [OUTPUT_DIR]" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---- Phase 1: 启动 RTSP 摄像头模拟 ----
echo "============================================"
echo "  Phase 1: Starting Camera Simulation"
echo "============================================"
bash "$ROOT_DIR/scripts/simulate_cameras.sh" "$VIDEO_DIR" --limit "$SOURCE_COUNT"

# ---- Phase 2: 启动 DeepStream 推理 ----
echo ""
echo "============================================"
echo "  Phase 2: Starting DeepStream Pipeline"
echo "============================================"

RUNTIME_DIR="$OUTPUT_DIR/.runtime"
mkdir -p "$RUNTIME_DIR/infer"

CONFIG_PATH="$RUNTIME_DIR/campus_surveillance.yaml"
JSONL_PATH="$OUTPUT_DIR/results.jsonl"
LOG_PATH="$OUTPUT_DIR/run.log"
PREVIEW_MP4="$OUTPUT_DIR/tiled_preview.mp4"

# 生成运行时配置
cat >"$CONFIG_PATH" <<YAML
app:
  app_name: campus-surveillance
  source_count: $SOURCE_COUNT
  enable_web: false

web:
  enabled: false

sources:
YAML
for i in $(seq 1 "$SOURCE_COUNT"); do
    printf '  - name: cam_%02d\n' "$i" >>"$CONFIG_PATH"
    printf '    kind: rtsp\n' >>"$CONFIG_PATH"
    printf '    uri: rtsp://127.0.0.1:%s/%s%d\n' "$RTSP_PORT" "$MOUNT_PREFIX" "$i" >>"$CONFIG_PATH"
    printf '    enabled: true\n' >>"$CONFIG_PATH"
done

cat >>"$CONFIG_PATH" <<YAML

logging:
  level: INFO
  file_path: $OUTPUT_DIR/app.log
  console: true

output:
  jsonl_path: $JSONL_PATH
  enable_jsonl: true

optimization:
  max_queue_size: 32
  fps_min: 5.0
  fps_max: 30.0
  enable_fps_control: true
  enable_backpressure: true

deepstream:
  batch_size: $SOURCE_COUNT
  batched_push_timeout_us: 40000
  inference_width: 640
  inference_height: 640
  enable_tracker: true
  enable_osd: true
  enable_tiler: true
  tiler_rows: 2
  tiler_columns: 4
  tiler_width: 1920
  tiler_height: 1080
  output_sink: file
  output_video_path: $PREVIEW_MP4
  model_engine_path: models/yolov8s.engine
  custom_lib_path: custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
  tracker_config_path: configs/deepstream/tracker_iou.yml
  infer_config_path: configs/deepstream/infer_primary_yolo_minimal.txt
  streammux_config_path: configs/deepstream/streammux.yaml
YAML

rm -f "$JSONL_PATH" "$LOG_PATH" "$PREVIEW_MP4"

echo "Config:  $CONFIG_PATH"
echo "JSONL:   $JSONL_PATH"
echo "Preview: $PREVIEW_MP4"
echo "Sources: $SOURCE_COUNT"
echo ""

source scripts/env.sh

PYTHONPATH=src python3 -m app.main \
    --config "$CONFIG_PATH" \
    --no-web \
    --confidence-threshold 0.25 \
    --runtime-dir "$RUNTIME_DIR/infer" \
    >"$LOG_PATH" 2>&1 &
DS_PID=$!

echo "DeepStream PID: $DS_PID"

# ---- Phase 3: 等待完成 ----
echo ""
echo "============================================"
echo "  Phase 3: Running (Ctrl+C to stop)"
echo "============================================"
echo ""
echo "DeepStream is processing $SOURCE_COUNT RTSP streams..."
echo "Press Ctrl+C to stop and generate reports."
echo ""

wait "$DS_PID" || true

# ---- Phase 4: 生成分析报告 ----
echo ""
echo "============================================"
echo "  Phase 4: Generating Reports"
echo "============================================"

if [ -f "$JSONL_PATH" ]; then
    echo "Generating multifile summary..."
    PYTHONPATH=scripts python3 -c "
from summarize_multifile_inproc import summarize
from pathlib import Path
summarize(Path('$JSONL_PATH'), Path('$OUTPUT_DIR'))
" 2>/dev/null || echo "  (summary generation skipped — run manually if needed)"

    echo "Running quality check..."
    PYTHONPATH=scripts python3 -c "
from check_multifile_inproc_outputs import check
from pathlib import Path
check(Path('$OUTPUT_DIR'))
" 2>/dev/null || echo "  (quality check skipped — run manually if needed)"
fi

echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
echo ""
echo "Output files:"
echo "  Tiled preview:  $PREVIEW_MP4"
echo "  JSONL results:  $JSONL_PATH"
echo "  Run log:        $LOG_PATH"
echo "  Summary:        $OUTPUT_DIR/multifile_summary.json"
echo "  Quality:        $OUTPUT_DIR/multifile_quality.json"
echo ""
echo "Launch Web UI to review:"
echo "  PYTHONPATH=src python3 scripts/preview_web.py --multifile-dir $OUTPUT_DIR"
echo ""
