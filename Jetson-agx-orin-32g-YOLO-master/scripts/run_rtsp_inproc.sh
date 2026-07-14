#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/project_paths.sh"

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
  RTSP_URIS=rtsp://127.0.0.1:8554/stream1,rtsp://127.0.0.1:8554/stream2
  OUTPUT_SINK=fake
  OUTPUT_URL=rtmp://127.0.0.1/live/stream
  OUTPUT_WIDTH=640
  OUTPUT_HEIGHT=640
  ENABLE_TILER=1
  TILER_ROWS=2
  TILER_COLUMNS=4
  TILER_WIDTH=1280
  TILER_HEIGHT=640
  CONFIDENCE_THRESHOLD=0.25
  RUN_SECONDS=40
  ENABLE_DROP_OLD_FRAMES=1
  ENABLE_HARDWARE_FALLBACK=1
  ENABLE_LAST_FRAME_KEEPALIVE=1
  LAST_FRAME_KEEPALIVE_TIMEOUT_MS=1000
  STALE_AFTER_SECONDS=5
  ENCODER_BITRATE=12000000

Examples:
  python3 scripts/serve_mp4_as_rtsp.py "$VIDEO_DIR" --limit 8

  scripts/run_rtsp_inproc.sh "$OUTPUT_ROOT/rtsp_inproc"
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

OUTPUT_DIR="${1:-$OUTPUT_ROOT/rtsp_inproc}"
SOURCE_COUNT="${SOURCE_COUNT:-8}"
RTSP_BASE="${RTSP_BASE:-rtsp://127.0.0.1:8554/stream}"
RTSP_URIS="${RTSP_URIS:-}"
OUTPUT_SINK="${OUTPUT_SINK:-fake}"
OUTPUT_URL="${OUTPUT_URL:-rtmp://127.0.0.1/live/stream}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-640}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-640}"
ENABLE_TILER="${ENABLE_TILER:-1}"
TILER_ROWS="${TILER_ROWS:-2}"
TILER_COLUMNS="${TILER_COLUMNS:-4}"
TILER_WIDTH="${TILER_WIDTH:-1280}"
TILER_HEIGHT="${TILER_HEIGHT:-640}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"
RUN_SECONDS="${RUN_SECONDS:-40}"
ENABLE_DROP_OLD_FRAMES="${ENABLE_DROP_OLD_FRAMES:-1}"
ENABLE_HARDWARE_FALLBACK="${ENABLE_HARDWARE_FALLBACK:-1}"
ENABLE_LAST_FRAME_KEEPALIVE="${ENABLE_LAST_FRAME_KEEPALIVE:-1}"
LAST_FRAME_KEEPALIVE_TIMEOUT_MS="${LAST_FRAME_KEEPALIVE_TIMEOUT_MS:-1000}"
STALE_AFTER_SECONDS="${STALE_AFTER_SECONDS:-5}"
ENCODER_BITRATE="${ENCODER_BITRATE:-12000000}"
if [ "$ENABLE_TILER" = "1" ]; then
    ENABLE_TILER_YAML=true
else
    ENABLE_TILER_YAML=false
fi
if [ "$ENABLE_DROP_OLD_FRAMES" = "1" ]; then
    ENABLE_DROP_OLD_FRAMES_YAML=true
else
    ENABLE_DROP_OLD_FRAMES_YAML=false
fi
if [ "$ENABLE_HARDWARE_FALLBACK" = "1" ]; then
    ENABLE_HARDWARE_FALLBACK_YAML=true
else
    ENABLE_HARDWARE_FALLBACK_YAML=false
fi
if [ "$ENABLE_LAST_FRAME_KEEPALIVE" = "1" ]; then
    ENABLE_LAST_FRAME_KEEPALIVE_YAML=true
else
    ENABLE_LAST_FRAME_KEEPALIVE_YAML=false
fi
RTSP_URI_ARRAY=()
if [ -n "$RTSP_URIS" ]; then
    IFS=',' read -r -a RTSP_URI_ARRAY <<< "$RTSP_URIS"
    SOURCE_COUNT="${#RTSP_URI_ARRAY[@]}"
fi

if ! [[ "$SOURCE_COUNT" =~ ^[0-9]+$ ]] || [ "$SOURCE_COUNT" -lt 1 ]; then
    echo "SOURCE_COUNT must be a positive integer: $SOURCE_COUNT" >&2
    exit 1
fi
if [ "$OUTPUT_SINK" != "fake" ] && [ "$OUTPUT_SINK" != "file" ] && [ "$OUTPUT_SINK" != "rtmp" ]; then
    if [ "$OUTPUT_SINK" != "rtsp" ]; then
        echo "OUTPUT_SINK must be one of: fake, file, rtmp, rtsp" >&2
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR/.runtime"

CONFIG_PATH="$OUTPUT_DIR/.runtime/app_rtsp_runtime.yaml"
JSONL_PATH="$OUTPUT_DIR/results.jsonl"
METRICS_JSONL_PATH="$OUTPUT_DIR/runtime_metrics.jsonl"
LOG_PATH="$OUTPUT_DIR/run.log"
RUNTIME_INFER_DIR="$OUTPUT_DIR/.runtime/infer"
RUN_METADATA_PATH="$OUTPUT_DIR/run_metadata.json"
SOURCE_STATUS_PATH="${SOURCE_STATUS_PATH:-.runtime/mediamtx_sim/source_status.json}"

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
    if [ "${#RTSP_URI_ARRAY[@]}" -gt 0 ]; then
        uri="${RTSP_URI_ARRAY[$((index - 1))]}"
    else
        uri="${RTSP_BASE}${index}"
    fi
    printf '  - name: rtsp_stream_%02d\n' "$index" >>"$CONFIG_PATH"
    printf '    kind: rtsp\n' >>"$CONFIG_PATH"
    printf '    uri: %s\n' "$uri" >>"$CONFIG_PATH"
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
  metrics_jsonl_path: $METRICS_JSONL_PATH
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
  enable_drop_old_frames: $ENABLE_DROP_OLD_FRAMES_YAML
  stale_after_seconds: $STALE_AFTER_SECONDS

deepstream:
  batch_size: $SOURCE_COUNT
  batched_push_timeout_us: 40000
  inference_width: $OUTPUT_WIDTH
  inference_height: $OUTPUT_HEIGHT
  enable_tiler: $ENABLE_TILER_YAML
  tiler_rows: $TILER_ROWS
  tiler_columns: $TILER_COLUMNS
  tiler_width: $TILER_WIDTH
  tiler_height: $TILER_HEIGHT
  enable_tracker: true
  enable_osd: true
  output_sink: $OUTPUT_SINK
  output_url: $OUTPUT_URL
  output_video_path: $OUTPUT_DIR/rtsp_preview.mp4
  model_engine_path: models/yolov8s.engine
  custom_lib_path: custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
  tracker_config_path: configs/deepstream/tracker_iou.yml
  infer_config_path: configs/deepstream/infer_primary_yolo_minimal.txt
  streammux_config_path: configs/deepstream/streammux.yaml
  enable_hardware_fallback: $ENABLE_HARDWARE_FALLBACK_YAML
  enable_last_frame_keepalive: $ENABLE_LAST_FRAME_KEEPALIVE_YAML
  last_frame_keepalive_timeout_ms: $LAST_FRAME_KEEPALIVE_TIMEOUT_MS
  encoder_bitrate: $ENCODER_BITRATE
YAML

rm -f "$JSONL_PATH" "$METRICS_JSONL_PATH" "$LOG_PATH" "$OUTPUT_DIR/rtsp_preview.mp4" "$RUN_METADATA_PATH" "$OUTPUT_DIR/source_status.json"

echo "== In-Process RTSP DeepStream Run =="
echo "Source count: $SOURCE_COUNT"
echo "RTSP base: $RTSP_BASE"
echo "Output directory: $OUTPUT_DIR"
echo "Output sink: $OUTPUT_SINK"
echo "Run seconds: $RUN_SECONDS"
echo "Tiler: $ENABLE_TILER (${TILER_ROWS}x${TILER_COLUMNS}, ${TILER_WIDTH}x${TILER_HEIGHT})"
echo "Drop old frames: $ENABLE_DROP_OLD_FRAMES"
echo "Hardware fallback: $ENABLE_HARDWARE_FALLBACK"
echo "Last-frame keepalive: $ENABLE_LAST_FRAME_KEEPALIVE (${LAST_FRAME_KEEPALIVE_TIMEOUT_MS}ms)"
echo "Encoder bitrate: $ENCODER_BITRATE"
echo "Runtime config: $CONFIG_PATH"
echo "JSONL: $JSONL_PATH"
echo "Metrics JSONL: $METRICS_JSONL_PATH"
echo ""

source scripts/env.sh

started_at="$(date -Iseconds)"
run_status="ok"
run_error=""
set +e
OUTPUT_SINK="$OUTPUT_SINK" \
OUTPUT_URL="$OUTPUT_URL" \
OUTPUT_VIDEO_PATH="$OUTPUT_DIR/rtsp_preview.mp4" \
RUNTIME_DIR="$RUNTIME_INFER_DIR" \
CONFIDENCE_THRESHOLD="$CONFIDENCE_THRESHOLD" \
RUN_SECONDS="$RUN_SECONDS" \
scripts/run_multistream.sh "$CONFIG_PATH" "$OUTPUT_DIR" \
    >"$LOG_PATH" 2>&1
app_exit=$?
set -e
finished_at="$(date -Iseconds)"

if [ "$app_exit" -ne 0 ]; then
    run_status="failed"
    run_error="app exited with code $app_exit"
fi

if [ -f "$SOURCE_STATUS_PATH" ]; then
    cp "$SOURCE_STATUS_PATH" "$OUTPUT_DIR/source_status.json"
fi

python3 - "$RUN_METADATA_PATH" <<PY
import json
import os
import sys

path = sys.argv[1]
source_count = int(os.environ.get("SOURCE_COUNT", "$SOURCE_COUNT"))
rtsp_base = os.environ.get("RTSP_BASE", "$RTSP_BASE")
payload = {
    "mode": "rtsp_inprocess",
    "status": "$run_status",
    "exit_code": $app_exit,
    "error": "$run_error",
    "started_at": "$started_at",
    "finished_at": "$finished_at",
    "source_count": source_count,
    "rtsp_base": rtsp_base,
    "run_seconds": float(os.environ.get("RUN_SECONDS", "$RUN_SECONDS")),
    "output_sink": os.environ.get("OUTPUT_SINK", "$OUTPUT_SINK"),
    "enable_tiler": os.environ.get("ENABLE_TILER", "$ENABLE_TILER") == "1",
    "tiler_rows": int(os.environ.get("TILER_ROWS", "$TILER_ROWS")),
    "tiler_columns": int(os.environ.get("TILER_COLUMNS", "$TILER_COLUMNS")),
    "tiler_width": int(os.environ.get("TILER_WIDTH", "$TILER_WIDTH")),
    "tiler_height": int(os.environ.get("TILER_HEIGHT", "$TILER_HEIGHT")),
    "enable_drop_old_frames": os.environ.get("ENABLE_DROP_OLD_FRAMES", "$ENABLE_DROP_OLD_FRAMES") == "1",
    "enable_hardware_fallback": os.environ.get("ENABLE_HARDWARE_FALLBACK", "$ENABLE_HARDWARE_FALLBACK") == "1",
    "enable_last_frame_keepalive": os.environ.get("ENABLE_LAST_FRAME_KEEPALIVE", "$ENABLE_LAST_FRAME_KEEPALIVE") == "1",
    "last_frame_keepalive_timeout_ms": int(os.environ.get("LAST_FRAME_KEEPALIVE_TIMEOUT_MS", "$LAST_FRAME_KEEPALIVE_TIMEOUT_MS")),
    "stale_after_seconds": float(os.environ.get("STALE_AFTER_SECONDS", "$STALE_AFTER_SECONDS")),
    "encoder_bitrate": int(os.environ.get("ENCODER_BITRATE", "$ENCODER_BITRATE")),
    "output_dir": "$OUTPUT_DIR",
    "output_video": "$OUTPUT_DIR/rtsp_preview.mp4",
    "output_jsonl": "$JSONL_PATH",
    "metrics_jsonl": "$METRICS_JSONL_PATH",
    "log_path": "$LOG_PATH",
    "runtime_config": "$CONFIG_PATH",
    "source_status_path": "$OUTPUT_DIR/source_status.json",
    "input_streams": [
        item for item in os.environ.get("RTSP_URIS", "").split(",") if item
    ] or [f"{rtsp_base}{index}" for index in range(1, source_count + 1)],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\\n")
PY

echo "In-process RTSP run completed."
echo "  JSONL: $JSONL_PATH"
echo "  Metrics JSONL: $METRICS_JSONL_PATH"
echo "  Log: $LOG_PATH"
echo "  Metadata: $RUN_METADATA_PATH"
exit "$app_exit"
