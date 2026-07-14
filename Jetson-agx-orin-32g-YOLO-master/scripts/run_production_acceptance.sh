#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/project_paths.sh"

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_production_acceptance.sh [OUTPUT_DIR]

Purpose:
  Current production acceptance entry for the no-real-camera phase:
  local MP4 files -> MediaMTX RTSP simulator -> one DeepStream pipeline ->
  JSONL/runtime metrics/quality summary -> dashboard.

Default:
  OUTPUT_DIR=$OUTPUT_ROOT/production_acceptance_latest

Important environment overrides:
  VIDEO_DIR=$VIDEO_DIR
  SOURCE_COUNT=8
  RTSP_PORT=8555
  OUTPUT_SINK=file
  RUN_SECONDS=40
  START_UI=1
  CHECK_RECOVERY=1
  MIN_FPS=0.5
  MIN_METRIC_SAMPLES=1
  MAX_STALE_COUNT=5
  ENABLE_TEGRASTATS=1
  ENABLE_INDIVIDUAL_OUTPUTS=1
  INDIVIDUAL_JOBS=1
  ENABLE_TILED_OUTPUT=0

Example:
  RUN_SECONDS=40 START_UI=1 scripts/run_production_acceptance.sh
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

OUTPUT_DIR="${1:-$OUTPUT_ROOT/production_acceptance_latest}"
SOURCE_COUNT="${SOURCE_COUNT:-8}"
RTSP_PORT="${RTSP_PORT:-8555}"
OUTPUT_SINK="${OUTPUT_SINK:-file}"
ENABLE_TILED_OUTPUT="${ENABLE_TILED_OUTPUT:-0}"
RUN_SECONDS="${RUN_SECONDS:-40}"
START_UI="${START_UI:-1}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8090}"
CHECK_RECOVERY="${CHECK_RECOVERY:-1}"
MIN_FPS="${MIN_FPS:-0.5}"
MIN_METRIC_SAMPLES="${MIN_METRIC_SAMPLES:-1}"
MAX_STALE_COUNT="${MAX_STALE_COUNT:-5}"
ENABLE_INDIVIDUAL_OUTPUTS="${ENABLE_INDIVIDUAL_OUTPUTS:-1}"
INDIVIDUAL_JOBS="${INDIVIDUAL_JOBS:-1}"

mkdir -p "$OUTPUT_DIR"

echo "== Production Acceptance =="
echo "Project root: $PROJECT_ROOT"
echo "Video dir: $VIDEO_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Source count: $SOURCE_COUNT"
echo "RTSP port: $RTSP_PORT"
echo "Output sink: $OUTPUT_SINK"
echo "Tiled output: $ENABLE_TILED_OUTPUT"
echo "Run seconds: $RUN_SECONDS"
echo "Start UI: $START_UI"
echo "Check recovery: $CHECK_RECOVERY"
echo "Min FPS: $MIN_FPS"
echo "Min metric samples: $MIN_METRIC_SAMPLES"
echo "Max stale count: $MAX_STALE_COUNT"
echo "Individual outputs: $ENABLE_INDIVIDUAL_OUTPUTS"
echo "Individual jobs: $INDIVIDUAL_JOBS"
echo ""

SOURCE_COUNT="$SOURCE_COUNT" \
RTSP_PORT="$RTSP_PORT" \
OUTPUT_SINK="$([ "$ENABLE_TILED_OUTPUT" = "1" ] && echo "$OUTPUT_SINK" || echo fake)" \
RUN_SECONDS="$RUN_SECONDS" \
START_UI=0 \
CHECK_RECOVERY="$CHECK_RECOVERY" \
MIN_FPS="$MIN_FPS" \
MIN_METRIC_SAMPLES="$MIN_METRIC_SAMPLES" \
MAX_STALE_COUNT="$MAX_STALE_COUNT" \
scripts/run_rtsp_acceptance.sh "$VIDEO_DIR" "$OUTPUT_DIR"

# Produce one independently encoded OSD video per source. The individual
# videos are the review artifacts; the tiled video is optional.
if [ "$ENABLE_INDIVIDUAL_OUTPUTS" = "1" ]; then
    echo ""
    echo "== Individual Stream Outputs =="
    SOURCE_COUNT="$SOURCE_COUNT" \
    INDIVIDUAL_JOBS="$INDIVIDUAL_JOBS" \
    RUN_SECONDS="$RUN_SECONDS" \
    scripts/run_rtsp_individual_outputs.sh \
        "$VIDEO_DIR" \
        "$OUTPUT_DIR/individual"
    echo "Individual output index: $OUTPUT_DIR/individual/individual_outputs.json"
fi

if [ "$START_UI" = "1" ]; then
    echo ""
    echo "Starting local dashboard..."
    echo "Open: http://$UI_HOST:$UI_PORT"
    python3 scripts/preview_web.py \
        --host "$UI_HOST" \
        --port "$UI_PORT" \
        --batch-dir "$OUTPUT_DIR/.no_batch" \
        --rtsp-dir "$OUTPUT_DIR" \
        --multifile-dir "$OUTPUT_DIR"
fi
