#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'USAGE'
Usage:
  scripts/legacy/person_analytics/run_multifile_inproc.sh INPUT_VIDEO_DIR [OUTPUT_DIR]

Purpose:
  Run multiple local MP4 files in one Python/DeepStream process through a
  single nvstreammux batch. This is the first step toward replacing
  multi-process batch runs with one in-process multi-source pipeline.

Optional environment overrides:
  VIDEO_GLOB=*.mp4
  SOURCE_COUNT=8
  OUTPUT_SINK=file
  ENABLE_OSD=1
  ENABLE_FPS_CONTROL=1
  ENABLE_BACKPRESSURE=1
  OUTPUT_URL=rtmp://127.0.0.1/live/stream
  OUTPUT_WIDTH=640
  OUTPUT_HEIGHT=640
  ENABLE_TILER=1
  TILER_ROWS=2
  TILER_COLUMNS=4
  TILER_WIDTH=1280
  TILER_HEIGHT=720
  CONFIDENCE_THRESHOLD=0.25
  ENCODER_BITRATE=12000000

Notes:
  - Default OUTPUT_SINK=file writes a single tiled MP4 preview plus JSONL.
  - Results are written to OUTPUT_DIR/results.jsonl.
  - Each record contains stream_id/source_id so the 8 inputs can be separated.

Examples:
  scripts/legacy/person_analytics/run_multifile_inproc.sh "$VIDEO_DIR" "$OUTPUT_ROOT/multifile_inproc"

  SOURCE_COUNT=4 scripts/legacy/person_analytics/run_multifile_inproc.sh "$VIDEO_DIR" "$OUTPUT_ROOT/multifile_4"
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-outputs/multifile_inproc}"
VIDEO_GLOB="${VIDEO_GLOB:-*.mp4}"
SOURCE_COUNT="${SOURCE_COUNT:-8}"
OUTPUT_SINK="${OUTPUT_SINK:-file}"
ENABLE_OSD="${ENABLE_OSD:-1}"
ENABLE_FPS_CONTROL="${ENABLE_FPS_CONTROL:-1}"
ENABLE_BACKPRESSURE="${ENABLE_BACKPRESSURE:-1}"
OUTPUT_URL="${OUTPUT_URL:-rtmp://127.0.0.1/live/stream}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-640}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-640}"
ENABLE_TILER="${ENABLE_TILER:-1}"
TILER_ROWS="${TILER_ROWS:-2}"
TILER_COLUMNS="${TILER_COLUMNS:-4}"
TILER_WIDTH="${TILER_WIDTH:-1280}"
TILER_HEIGHT="${TILER_HEIGHT:-720}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"
ENCODER_BITRATE="${ENCODER_BITRATE:-12000000}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Input video directory not found: $INPUT_DIR" >&2
    exit 1
fi
if ! [[ "$SOURCE_COUNT" =~ ^[0-9]+$ ]] || [ "$SOURCE_COUNT" -lt 1 ]; then
    echo "SOURCE_COUNT must be a positive integer: $SOURCE_COUNT" >&2
    exit 1
fi
if [ "$OUTPUT_SINK" != "fake" ] && [ "$OUTPUT_SINK" != "file" ] && [ "$OUTPUT_SINK" != "rtmp" ] && [ "$OUTPUT_SINK" != "rtsp" ]; then
    echo "OUTPUT_SINK must be one of: fake, file, rtmp, rtsp" >&2
    exit 1
fi
if [ "$ENABLE_OSD" != "0" ] && [ "$ENABLE_OSD" != "1" ]; then
    echo "ENABLE_OSD must be 0 or 1: $ENABLE_OSD" >&2
    exit 1
fi
if [ "$ENABLE_FPS_CONTROL" != "0" ] && [ "$ENABLE_FPS_CONTROL" != "1" ]; then exit 1; fi
if [ "$ENABLE_BACKPRESSURE" != "0" ] && [ "$ENABLE_BACKPRESSURE" != "1" ]; then exit 1; fi

mkdir -p "$OUTPUT_DIR/.runtime"

mapfile -t VIDEOS < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "$VIDEO_GLOB" | sort | head -n "$SOURCE_COUNT")
if [ "${#VIDEOS[@]}" -ne "$SOURCE_COUNT" ]; then
    echo "Expected $SOURCE_COUNT videos matching $VIDEO_GLOB in $INPUT_DIR, found ${#VIDEOS[@]}" >&2
    exit 1
fi
for video in "${VIDEOS[@]}"; do
    if [ ! -s "$video" ]; then
        echo "Input video is missing or empty: $video" >&2
        exit 1
    fi
done

CONFIG_PATH="$OUTPUT_DIR/.runtime/app_multifile_runtime.yaml"
JSONL_PATH="$OUTPUT_DIR/results.jsonl"
LOG_PATH="$OUTPUT_DIR/run.log"
RUN_METADATA_PATH="$OUTPUT_DIR/run_metadata.json"
RUNTIME_INFER_DIR="$OUTPUT_DIR/.runtime/infer"
SUMMARY_PATH="$OUTPUT_DIR/multifile_summary.json"
QUALITY_PATH="$OUTPUT_DIR/multifile_quality.json"

cat >"$CONFIG_PATH" <<YAML
app:
  app_name: deepstream-yolov8-multifile-inproc
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
for video in "${VIDEOS[@]}"; do
    printf '  - name: local_video_%02d\n' "$index" >>"$CONFIG_PATH"
    printf '    kind: file\n' >>"$CONFIG_PATH"
    printf '    uri: %s\n' "$video" >>"$CONFIG_PATH"
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
  enable_fps_control: $([ "$ENABLE_FPS_CONTROL" = "1" ] && echo true || echo false)
  enable_backpressure: $([ "$ENABLE_BACKPRESSURE" = "1" ] && echo true || echo false)

deepstream:
  batch_size: $SOURCE_COUNT
  batched_push_timeout_us: 40000
  inference_width: $OUTPUT_WIDTH
  inference_height: $OUTPUT_HEIGHT
  enable_tracker: true
  enable_osd: $([ "$ENABLE_OSD" = "1" ] && echo true || echo false)
  enable_tiler: $([ "$ENABLE_TILER" = "1" ] && echo true || echo false)
  tiler_rows: $TILER_ROWS
  tiler_columns: $TILER_COLUMNS
  tiler_width: $TILER_WIDTH
  tiler_height: $TILER_HEIGHT
  output_sink: $OUTPUT_SINK
  output_url: $OUTPUT_URL
  output_video_path: $OUTPUT_DIR/multifile_preview.mp4
  model_engine_path: models/yolov8s.engine
  custom_lib_path: custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
  tracker_config_path: configs/deepstream/tracker_iou.yml
  infer_config_path: configs/deepstream/infer_primary_yolo_minimal.txt
  streammux_config_path: configs/deepstream/streammux.yaml
  encoder_bitrate: $ENCODER_BITRATE
YAML

rm -f "$JSONL_PATH" "$LOG_PATH" "$RUN_METADATA_PATH" "$OUTPUT_DIR/multifile_preview.mp4" "$SUMMARY_PATH" "$QUALITY_PATH"

echo "== In-Process Multi-File DeepStream Run =="
echo "Input directory: $INPUT_DIR"
echo "Video glob: $VIDEO_GLOB"
echo "Source count: $SOURCE_COUNT"
echo "Output directory: $OUTPUT_DIR"
echo "Output sink: $OUTPUT_SINK"
echo "OSD: $ENABLE_OSD"
echo "FPS control: $ENABLE_FPS_CONTROL"
echo "Backpressure: $ENABLE_BACKPRESSURE"
echo "Tiler: $ENABLE_TILER (${TILER_ROWS}x${TILER_COLUMNS}, ${TILER_WIDTH}x${TILER_HEIGHT})"
echo "Runtime config: $CONFIG_PATH"
echo "JSONL: $JSONL_PATH"
echo ""
printf '%s\n' "${VIDEOS[@]}" | nl -w2 -s'. '
echo ""

source scripts/deploy/env.sh

started_at="$(date --iso-8601=seconds)"
set +e
OUTPUT_SINK="$OUTPUT_SINK" \
OUTPUT_URL="$OUTPUT_URL" \
OUTPUT_VIDEO_PATH="$OUTPUT_DIR/multifile_preview.mp4" \
RUNTIME_DIR="$RUNTIME_INFER_DIR" \
CONFIDENCE_THRESHOLD="$CONFIDENCE_THRESHOLD" \
RUN_SECONDS=0 \
scripts/deploy/run_multistream.sh "$CONFIG_PATH" "$OUTPUT_DIR" \
    >"$LOG_PATH" 2>&1
app_exit=$?
set -e
finished_at="$(date --iso-8601=seconds)"

if [ "$app_exit" -eq 0 ]; then
    run_status="ok"
    run_error=""
else
    run_status="failed"
    run_error="app.main exited with $app_exit"
fi

python3 - "$RUN_METADATA_PATH" "$run_status" "$app_exit" "$started_at" "$finished_at" "$run_error" "$LOG_PATH" "$SOURCE_COUNT" "${VIDEOS[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "started_at": sys.argv[4],
    "finished_at": sys.argv[5],
    "error": sys.argv[6],
    "log_path": sys.argv[7],
    "source_count": int(sys.argv[8]),
    "input_videos": sys.argv[9:],
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python3 scripts/legacy/person_analytics/summarize_multifile_inproc.py \
    "$OUTPUT_DIR" \
    "$SUMMARY_PATH" \
    --expected-stream-count "$SOURCE_COUNT"

quality_exit=0
python3 scripts/legacy/person_analytics/check_multifile_inproc_outputs.py \
    "$SUMMARY_PATH" \
    "$QUALITY_PATH" || quality_exit=$?

echo "In-process multi-file run completed."
if [ "$OUTPUT_SINK" = "file" ]; then
    echo "  Tiled video: $OUTPUT_DIR/multifile_preview.mp4"
fi
echo "  JSONL: $JSONL_PATH"
echo "  Summary: $SUMMARY_PATH"
echo "  Quality: $QUALITY_PATH"
echo "  Log: $LOG_PATH"
echo "  Run metadata: $RUN_METADATA_PATH"
echo ""
echo "Quick checks:"
echo "  wc -l $JSONL_PATH"
echo "  tail -5 $JSONL_PATH"

if [ "$app_exit" -ne 0 ]; then
    exit "$app_exit"
fi
exit "$quality_exit"
