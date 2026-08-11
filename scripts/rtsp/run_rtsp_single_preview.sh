#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/deploy/project_paths.sh"

usage() {
    cat <<'USAGE'
Usage:
  scripts/rtsp/run_rtsp_single_preview.sh [INPUT_VIDEO_OR_DIR] [OUTPUT_DIR]

Purpose:
  Diagnose RTSP input stability with one stream only:
  local MP4 -> MediaMTX RTSP simulator -> one DeepStream source -> OSD MP4.

Defaults:
  INPUT_VIDEO_OR_DIR=$VIDEO_DIR
  OUTPUT_DIR=$OUTPUT_ROOT/rtsp_single_preview

Environment:
  RTSP_PORT=8556
  RTSP_HOST=127.0.0.1
  MOUNT_PREFIX=stream
  RUN_SECONDS=60
  OUTPUT_SINK=file
  OUTPUT_WIDTH=640
  OUTPUT_HEIGHT=640
  ENCODER_BITRATE=8000000
  CONFIDENCE_THRESHOLD=0.25
  SIM_TRANSCODE=0
  SIM_TRANSCODE_FPS=15
  SIM_TRANSCODE_BITRATE=2500k
  SIM_TRANSCODE_GOP=30

Examples:
  scripts/rtsp/run_rtsp_single_preview.sh "$VIDEO_DIR"

  SIM_TRANSCODE=1 scripts/rtsp/run_rtsp_single_preview.sh \
    "$VIDEO_DIR/1.mp4" \
    outputs/rtsp_single_transcode
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

INPUT="${1:-$VIDEO_DIR}"
OUTPUT_DIR="${2:-$OUTPUT_ROOT/rtsp_single_preview}"
RTSP_PORT="${RTSP_PORT:-8556}"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
MOUNT_PREFIX="${MOUNT_PREFIX:-stream}"
RUN_SECONDS="${RUN_SECONDS:-60}"
OUTPUT_SINK="${OUTPUT_SINK:-file}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-640}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-640}"
ENCODER_BITRATE="${ENCODER_BITRATE:-8000000}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"

mkdir -p "$OUTPUT_DIR"

echo "== RTSP Single Preview Diagnostic =="
echo "Input: $INPUT"
echo "Output directory: $OUTPUT_DIR"
echo "RTSP: rtsp://$RTSP_HOST:$RTSP_PORT/${MOUNT_PREFIX}1"
echo "Run seconds: $RUN_SECONDS"
echo "Output size: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "Encoder bitrate: $ENCODER_BITRATE"
echo "Simulator transcode: ${SIM_TRANSCODE:-0}"
echo ""

RTSP_PORT="$RTSP_PORT" \
RTSP_HOST="$RTSP_HOST" \
MOUNT_PREFIX="$MOUNT_PREFIX" \
RUNTIME_DIR=".runtime/mediamtx_single" \
scripts/rtsp/simulate_cameras.sh "$INPUT" --limit 1

sleep 3

SOURCE_COUNT=1 \
RTSP_BASE="rtsp://$RTSP_HOST:$RTSP_PORT/$MOUNT_PREFIX" \
OUTPUT_SINK="$OUTPUT_SINK" \
OUTPUT_WIDTH="$OUTPUT_WIDTH" \
OUTPUT_HEIGHT="$OUTPUT_HEIGHT" \
ENABLE_TILER=0 \
RUN_SECONDS="$RUN_SECONDS" \
CONFIDENCE_THRESHOLD="$CONFIDENCE_THRESHOLD" \
SOURCE_STATUS_PATH=".runtime/mediamtx_single/source_status.json" \
ENCODER_BITRATE="$ENCODER_BITRATE" \
scripts/rtsp/run_rtsp_inproc.sh "$OUTPUT_DIR"

if [ "$OUTPUT_SINK" = "file" ] && [ -f "$OUTPUT_DIR/rtsp_preview.mp4" ]; then
    mv "$OUTPUT_DIR/rtsp_preview.mp4" "$OUTPUT_DIR/rtsp_single_preview.mp4"
fi

if [ -f "$OUTPUT_DIR/source_status.json" ]; then
    cp "$OUTPUT_DIR/source_status.json" "$OUTPUT_DIR/source_status.single.json" || true
fi

echo ""
echo "Single RTSP preview completed."
echo "  Video: $OUTPUT_DIR/rtsp_single_preview.mp4"
echo "  JSONL: $OUTPUT_DIR/results.jsonl"
echo "  Log: $OUTPUT_DIR/run.log"
echo "  MediaMTX log: .runtime/mediamtx_single/mediamtx.log"
echo ""
echo "Checks:"
echo "  gst-discoverer-1.0 $OUTPUT_DIR/rtsp_single_preview.mp4"
echo "  grep -E \"reader is too slow|discarding|RTP packets are too big|ERROR|Traceback\" .runtime/mediamtx_single/mediamtx.log | tail -80"
