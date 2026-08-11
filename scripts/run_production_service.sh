#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/project_paths.sh"

SERVICE_OUTPUT_DIR="${SERVICE_OUTPUT_DIR:-$OUTPUT_ROOT/campus_surveillance}"
RTSP_PORT="${RTSP_PORT:-8555}"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
MOUNT_PREFIX="${MOUNT_PREFIX:-stream}"
SOURCE_COUNT="${SOURCE_COUNT:-8}"
OUTPUT_SINK="${OUTPUT_SINK:-fake}"
RUN_SECONDS="${RUN_SECONDS:-0}"
START_SIMULATOR="${START_SIMULATOR:-1}"
START_UI="${START_UI:-1}"
UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-8090}"
RTSP_BASE="${RTSP_BASE:-rtsp://$RTSP_HOST:$RTSP_PORT/$MOUNT_PREFIX}"

mkdir -p "$SERVICE_OUTPUT_DIR" "$LOG_ROOT" "$RUNTIME_ROOT"

UI_PID=""

cleanup() {
    if [ -n "$UI_PID" ] && kill -0 "$UI_PID" 2>/dev/null; then
        kill "$UI_PID" 2>/dev/null || true
        wait "$UI_PID" 2>/dev/null || true
    fi
    if [ "$START_SIMULATOR" = "1" ]; then
        RUNTIME_DIR="$RUNTIME_ROOT/mediamtx_sim" scripts/simulate_cameras.sh --stop 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "== Campus Surveillance Production Service =="
echo "Project root: $PROJECT_ROOT"
echo "Video dir: $VIDEO_DIR"
echo "Output dir: $SERVICE_OUTPUT_DIR"
echo "Runtime root: $RUNTIME_ROOT"
echo "Log root: $LOG_ROOT"
echo "RTSP base: $RTSP_BASE"
echo "Source count: $SOURCE_COUNT"
echo "Output sink: $OUTPUT_SINK"
echo "Run seconds: $RUN_SECONDS"
echo ""

if [ "$START_SIMULATOR" = "1" ]; then
    RUNTIME_DIR="$RUNTIME_ROOT/mediamtx_sim" \
    RTSP_PORT="$RTSP_PORT" \
    RTSP_HOST="$RTSP_HOST" \
    MOUNT_PREFIX="$MOUNT_PREFIX" \
    scripts/simulate_cameras.sh "$VIDEO_DIR" --limit "$SOURCE_COUNT"
    sleep 3
fi

if [ "$START_UI" = "1" ]; then
    python3 scripts/preview_web.py \
        --host "$UI_HOST" \
        --port "$UI_PORT" \
        --batch-dir "$OUTPUT_ROOT/batch" \
        --rtsp-dir "$SERVICE_OUTPUT_DIR" \
        --multifile-dir "$SERVICE_OUTPUT_DIR" &
    UI_PID=$!
    echo "Dashboard PID: $UI_PID"
    echo "Dashboard URL: http://$UI_HOST:$UI_PORT"
fi

SOURCE_COUNT="$SOURCE_COUNT" \
RTSP_BASE="$RTSP_BASE" \
OUTPUT_SINK="$OUTPUT_SINK" \
RUN_SECONDS="$RUN_SECONDS" \
SOURCE_STATUS_PATH="$RUNTIME_ROOT/mediamtx_sim/source_status.json" \
scripts/run_rtsp_inproc.sh "$SERVICE_OUTPUT_DIR"
