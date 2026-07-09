#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_rtsp_inproc.sh [OUTPUT_DIR]

Purpose:
  Pull local RTSP streams into one Python/DeepStream process and batch them
  with one nvstreammux. Use scripts/serve_mp4_as_rtsp.py first when you only
  have local MP4 files.

Optional environment overrides:
  SOURCE_COUNT=8
  RTSP_BASE=rtsp://127.0.0.1:8554/stream
  OUTPUT_SINK=fake
  OUTPUT_WIDTH=640
  OUTPUT_HEIGHT=640
  CONFIDENCE_THRESHOLD=0.25

Examples:
  python3 scripts/serve_mp4_as_rtsp.py /home/nvidia/Desktop/YOLO/video --limit 8

  scripts/run_rtsp_inproc.sh outputs/rtsp_inproc
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

OUTPUT_DIR="${1:-outputs/rtsp_inproc}"
SOURCE_COUNT="${SOURCE_COUNT:-8}"
RTSP_BASE="${RTSP_BASE:-rtsp://127.0.0.1:8554/stream}"
OUTPUT_SINK="${OUTPUT_SINK:-fake}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-640}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-640}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"

if ! [[ "$SOURCE_COUNT" =~ ^[0-9]+$ ]] || [ "$SOURCE_COUNT" -lt 1 ]; then
    echo "SOURCE_COUNT must be a positive integer: $SOURCE_COUNT" >&2
    exit 1
fi
if [ "$OUTPUT_SINK" != "fake" ] && [ "$OUTPUT_SINK" != "file" ] && [ "$OUTPUT_SINK" != "rtmp" ]; then
    echo "OUTPUT_SINK must be one of: fake, file, rtmp" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/.runtime"

CONFIG_PATH="$OUTPUT_DIR/.runtime/app_rtsp_runtime.yaml"
JSONL_PATH="$OUTPUT_DIR/results.jsonl"
LOG_PATH="$OUTPUT_DIR/run.log"
RUNTIME_INFER_DIR="$OUTPUT_DIR/.runtime/infer"

cat >"$CONFIG_PATH" <<YAML
app:
  app_name: deepstream-yolov8-rtsp-inproc
  source_count: $SOURCE_COUNT
  enable_web: false

web:
  enabled: false
  host: 127.0.0.1
  port: 8080
  enable_status_api: true
  enable_debug_api: true
  enable_logs_api: true
  refresh_interval_ms: 1000
  log_buffer_size: 200

sources:
YAML

index=1
while [ "$index" -le "$SOURCE_COUNT" ]; do
    printf '  - name: rtsp_stream_%02d\n' "$index" >>"$CONFIG_PATH"
    printf '    kind: rtsp\n' >>"$CONFIG_PATH"
    printf '    uri: %s%d\n' "$RTSP_BASE" "$index" >>"$CONFIG_PATH"
    printf '    enabled: true\n' >>"$CONFIG_PATH"
    index=$((index + 1))
done

cat >>"$CONFIG_PATH" <<YAML

logging:
  level: INFO
  file_path: $OUTPUT_DIR/app.log
  console: true

output:
  jsonl_path: $JSONL_PATH
  enable_jsonl: true
  enable_mqtt: false
  enable_kafka: false
  mqtt_host: 127.0.0.1
  mqtt_port: 1883
  mqtt_topic: deepstream/results

optimization:
  max_queue_size: 32
  fps_min: 5.0
  fps_max: 30.0
  enable_fps_control: false
  enable_backpressure: true

deepstream:
  batch_size: $SOURCE_COUNT
  batched_push_timeout_us: 40000
  inference_width: $OUTPUT_WIDTH
  inference_height: $OUTPUT_HEIGHT
  enable_tracker: true
  enable_osd: true
  output_sink: $OUTPUT_SINK
  output_video_path: $OUTPUT_DIR/rtsp_preview.mp4
  model_engine_path: models/yolov8s.engine
  custom_lib_path: custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
  tracker_config_path: configs/deepstream/tracker_iou.yml
  infer_config_path: configs/deepstream/infer_primary_yolo_minimal.txt
  streammux_config_path: configs/deepstream/streammux.yaml
YAML

rm -f "$JSONL_PATH" "$LOG_PATH" "$OUTPUT_DIR/rtsp_preview.mp4"

echo "== In-Process RTSP DeepStream Run =="
echo "Source count: $SOURCE_COUNT"
echo "RTSP base: $RTSP_BASE"
echo "Output directory: $OUTPUT_DIR"
echo "Output sink: $OUTPUT_SINK"
echo "Runtime config: $CONFIG_PATH"
echo "JSONL: $JSONL_PATH"
echo ""

source scripts/env.sh

PYTHONPATH=src python3 -m app.main \
    --config "$CONFIG_PATH" \
    --no-web \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --runtime-dir "$RUNTIME_INFER_DIR" \
    >"$LOG_PATH" 2>&1

echo "In-process RTSP run completed."
echo "  JSONL: $JSONL_PATH"
echo "  Log: $LOG_PATH"
