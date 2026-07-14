#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

RTSP_PORT="${RTSP_PORT:-8554}"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
MOUNT_PREFIX="${MOUNT_PREFIX:-stream}"
VIDEO_GLOB="${VIDEO_GLOB:-*.mp4}"
RUNTIME_DIR="${RUNTIME_DIR:-.runtime/mediamtx_sim}"
SIM_WRITE_QUEUE_SIZE="${SIM_WRITE_QUEUE_SIZE:-8192}"
SIM_RTSP_PKT_SIZE="${SIM_RTSP_PKT_SIZE:-1200}"
SIM_TRANSCODE="${SIM_TRANSCODE:-0}"
SIM_TRANSCODE_FPS="${SIM_TRANSCODE_FPS:-15}"
SIM_TRANSCODE_BITRATE="${SIM_TRANSCODE_BITRATE:-2500k}"
SIM_TRANSCODE_GOP="${SIM_TRANSCODE_GOP:-30}"
SIM_ENABLE_RTMP="${SIM_ENABLE_RTMP:-0}"

usage() {
    cat <<'EOF'
Usage:
  scripts/simulate_cameras.sh VIDEO_DIR [--limit N]    start MediaMTX + FFmpeg camera simulator
  scripts/simulate_cameras.sh --stop                    stop simulator
  scripts/simulate_cameras.sh --status                  print source_status.json

Environment:
  RTSP_PORT=8554
  RTSP_HOST=127.0.0.1
  MOUNT_PREFIX=stream
  VIDEO_GLOB=*.mp4
  RUNTIME_DIR=.runtime/mediamtx_sim
  SIM_WRITE_QUEUE_SIZE=8192
  SIM_RTSP_PKT_SIZE=1200
  SIM_TRANSCODE=0
  SIM_TRANSCODE_FPS=15
  SIM_TRANSCODE_BITRATE=2500k
  SIM_TRANSCODE_GOP=30

Examples:
  scripts/simulate_cameras.sh "$VIDEO_DIR"
  scripts/simulate_cameras.sh "$VIDEO_DIR" --limit 4
  scripts/simulate_cameras.sh --status
  scripts/simulate_cameras.sh --stop
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "${1:-}" = "--stop" ]; then
    python3 scripts/manage_mediamtx_sim.py stop --runtime-dir "$RUNTIME_DIR"
    exit 0
fi

if [ "${1:-}" = "--status" ]; then
    python3 scripts/manage_mediamtx_sim.py status --runtime-dir "$RUNTIME_DIR"
    exit 0
fi

VIDEO_DIR="${1:-}"
if [ -z "$VIDEO_DIR" ]; then
    echo "ERROR: VIDEO_DIR is required." >&2
    usage
    exit 1
fi

LIMIT=8
if [ "${2:-}" = "--limit" ] && [ -n "${3:-}" ]; then
    LIMIT="$3"
fi

extra_args=()
if [ "$SIM_TRANSCODE" = "1" ]; then
    extra_args+=(--transcode)
fi
if [ "$SIM_ENABLE_RTMP" = "1" ]; then
    extra_args+=(--enable-rtmp)
fi

python3 scripts/manage_mediamtx_sim.py start "$VIDEO_DIR" \
    --runtime-dir "$RUNTIME_DIR" \
    --host "$RTSP_HOST" \
    --rtsp-port "$RTSP_PORT" \
    --glob "$VIDEO_GLOB" \
    --limit "$LIMIT" \
    --mount-prefix "$MOUNT_PREFIX" \
    --write-queue-size "$SIM_WRITE_QUEUE_SIZE" \
    --rtsp-pkt-size "$SIM_RTSP_PKT_SIZE" \
    --transcode-fps "$SIM_TRANSCODE_FPS" \
    --transcode-bitrate "$SIM_TRANSCODE_BITRATE" \
    --transcode-gop "$SIM_TRANSCODE_GOP" \
    "${extra_args[@]}" \
    --force
