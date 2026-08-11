#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/deploy/project_paths.sh"

usage() {
    cat <<'USAGE'
Usage:
  scripts/rtsp/run_rtsp_acceptance.sh [INPUT_VIDEO_DIR] [OUTPUT_DIR]

Default:
  INPUT_VIDEO_DIR=$VIDEO_DIR
  OUTPUT_DIR=$OUTPUT_ROOT/rtsp_acceptance_latest

Optional environment overrides:
  SOURCE_COUNT=8
  RTSP_PORT=8555
  RTSP_HOST=127.0.0.1
  MOUNT_PREFIX=stream
  OUTPUT_SINK=file
  RUN_SECONDS=40
  CONFIDENCE_THRESHOLD=0.25
  START_UI=1
  UI_HOST=127.0.0.1
  UI_PORT=8090
  MIN_FPS=0.5
  MIN_METRIC_SAMPLES=1
  MAX_STALE_COUNT=5
  REQUIRE_PERSON=0
  CHECK_RECOVERY=1
  RECOVERY_STREAM_ID=stream1
  ENABLE_DROP_OLD_FRAMES=1
  ENABLE_HARDWARE_FALLBACK=1
  ENABLE_LAST_FRAME_KEEPALIVE=1
  LAST_FRAME_KEEPALIVE_TIMEOUT_MS=1000
  STALE_AFTER_SECONDS=5

Examples:
  scripts/rtsp/run_rtsp_acceptance.sh

  OUTPUT_SINK=fake RUN_SECONDS=40 START_UI=0 \
    scripts/rtsp/run_rtsp_acceptance.sh "$VIDEO_DIR" "$OUTPUT_ROOT/rtsp_acceptance_fake"
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

INPUT_DIR="${1:-$VIDEO_DIR}"
OUTPUT_DIR="${2:-$OUTPUT_ROOT/rtsp_acceptance_latest}"
SOURCE_COUNT="${SOURCE_COUNT:-8}"
RTSP_PORT="${RTSP_PORT:-8555}"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
MOUNT_PREFIX="${MOUNT_PREFIX:-stream}"
OUTPUT_SINK="${OUTPUT_SINK:-file}"
RUN_SECONDS="${RUN_SECONDS:-40}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"
START_UI="${START_UI:-1}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8090}"
MIN_FPS="${MIN_FPS:-0.5}"
MIN_METRIC_SAMPLES="${MIN_METRIC_SAMPLES:-1}"
MAX_STALE_COUNT="${MAX_STALE_COUNT:-5}"
REQUIRE_PERSON="${REQUIRE_PERSON:-0}"
CHECK_RECOVERY="${CHECK_RECOVERY:-1}"
RECOVERY_STREAM_ID="${RECOVERY_STREAM_ID:-stream1}"
ENABLE_DROP_OLD_FRAMES="${ENABLE_DROP_OLD_FRAMES:-1}"
ENABLE_HARDWARE_FALLBACK="${ENABLE_HARDWARE_FALLBACK:-1}"
ENABLE_LAST_FRAME_KEEPALIVE="${ENABLE_LAST_FRAME_KEEPALIVE:-1}"
LAST_FRAME_KEEPALIVE_TIMEOUT_MS="${LAST_FRAME_KEEPALIVE_TIMEOUT_MS:-1000}"
STALE_AFTER_SECONDS="${STALE_AFTER_SECONDS:-5}"

RTSP_BASE="rtsp://$RTSP_HOST:$RTSP_PORT/$MOUNT_PREFIX"
SUMMARY_PATH="$OUTPUT_DIR/rtsp_summary.json"
QUALITY_PATH="$OUTPUT_DIR/rtsp_quality.json"

echo "== RTSP In-Process Acceptance Flow =="
echo "Input video directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Source count: $SOURCE_COUNT"
echo "RTSP base: $RTSP_BASE"
echo "Output sink: $OUTPUT_SINK"
echo "Run seconds: $RUN_SECONDS"
echo "UI: http://$UI_HOST:$UI_PORT"
echo "Recovery check: $CHECK_RECOVERY ($RECOVERY_STREAM_ID)"
echo "Drop old frames: $ENABLE_DROP_OLD_FRAMES"
echo "Hardware fallback: $ENABLE_HARDWARE_FALLBACK"
echo ""

RTSP_PORT="$RTSP_PORT" \
RTSP_HOST="$RTSP_HOST" \
MOUNT_PREFIX="$MOUNT_PREFIX" \
scripts/rtsp/simulate_cameras.sh "$INPUT_DIR" --limit "$SOURCE_COUNT"
sim_exit=$?
if [ "$sim_exit" -ne 0 ]; then
    echo "Failed to start RTSP simulator: $sim_exit" >&2
    exit "$sim_exit"
fi

sleep 3

recovery_exit=0
if [ "$CHECK_RECOVERY" = "1" ]; then
    python3 scripts/rtsp/check_rtsp_recovery.py \
        --runtime-dir .runtime/mediamtx_sim \
        --stream-id "$RECOVERY_STREAM_ID" \
        --output-json "$OUTPUT_DIR/rtsp_recovery_check.json" || recovery_exit=$?
    sleep 3
fi

SOURCE_COUNT="$SOURCE_COUNT" \
RTSP_BASE="$RTSP_BASE" \
OUTPUT_SINK="$OUTPUT_SINK" \
RUN_SECONDS="$RUN_SECONDS" \
CONFIDENCE_THRESHOLD="$CONFIDENCE_THRESHOLD" \
SOURCE_STATUS_PATH=".runtime/mediamtx_sim/source_status.json" \
ENABLE_DROP_OLD_FRAMES="$ENABLE_DROP_OLD_FRAMES" \
ENABLE_HARDWARE_FALLBACK="$ENABLE_HARDWARE_FALLBACK" \
ENABLE_LAST_FRAME_KEEPALIVE="$ENABLE_LAST_FRAME_KEEPALIVE" \
LAST_FRAME_KEEPALIVE_TIMEOUT_MS="$LAST_FRAME_KEEPALIVE_TIMEOUT_MS" \
STALE_AFTER_SECONDS="$STALE_AFTER_SECONDS" \
scripts/rtsp/run_rtsp_inproc.sh "$OUTPUT_DIR"
run_exit=$?

python3 scripts/rtsp/summarize_rtsp_inproc.py "$OUTPUT_DIR" "$SUMMARY_PATH" \
    --expected-stream-count "$SOURCE_COUNT"

quality_args=(
    "$SUMMARY_PATH"
    "$QUALITY_PATH"
    "--min-fps"
    "$MIN_FPS"
    "--min-metric-samples"
    "$MIN_METRIC_SAMPLES"
    "--max-stale-count"
    "$MAX_STALE_COUNT"
)
if [ "$REQUIRE_PERSON" = "1" ]; then
    quality_args+=("--require-person")
fi
python3 scripts/rtsp/check_rtsp_inproc_outputs.py "${quality_args[@]}"
quality_exit=$?

echo ""
echo "== RTSP Acceptance Outputs =="
echo "RTSP summary: $SUMMARY_PATH"
echo "RTSP quality: $QUALITY_PATH"
echo "RTSP JSONL: $OUTPUT_DIR/results.jsonl"
echo "RTSP preview: $OUTPUT_DIR/rtsp_preview.mp4"
echo "Run log: $OUTPUT_DIR/run.log"
echo "Source status: $OUTPUT_DIR/source_status.json"
echo "Recovery check: $OUTPUT_DIR/rtsp_recovery_check.json"

if [ "$run_exit" -ne 0 ]; then
    echo "RTSP pipeline finished with failures: $run_exit" >&2
fi
if [ "$quality_exit" -ne 0 ]; then
    echo "RTSP quality check finished with failures: $quality_exit" >&2
fi
if [ "$recovery_exit" -ne 0 ]; then
    echo "RTSP recovery check finished with failures: $recovery_exit" >&2
fi

if [ "$START_UI" = "1" ]; then
    echo ""
    echo "Starting local dashboard..."
    echo "Open: http://$UI_HOST:$UI_PORT"
    echo "Press Ctrl+C to stop the dashboard."
    python3 scripts/tools/preview_web.py \
        --host "$UI_HOST" \
        --port "$UI_PORT" \
        --batch-dir "$OUTPUT_DIR/.no_batch" \
        --rtsp-dir "$OUTPUT_DIR" \
        --multifile-dir "$OUTPUT_DIR"
else
    echo ""
    echo "UI not started because START_UI=$START_UI"
    echo "Start manually with:"
    echo "python3 scripts/tools/preview_web.py --host $UI_HOST --port $UI_PORT --rtsp-dir $OUTPUT_DIR --multifile-dir $OUTPUT_DIR"
fi

exit "$quality_exit"
