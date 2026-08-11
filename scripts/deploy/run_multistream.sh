#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'USAGE'
Usage:
  scripts/deploy/run_multistream.sh CONFIG_PATH [OUTPUT_DIR]

Environment overrides:
  OUTPUT_SINK=fake|file|rtmp|rtsp
  OUTPUT_URL=rtmp://127.0.0.1/live/stream
  RUN_SECONDS=0
  START_UI=0

The YAML config owns the source list. The same application pipeline handles
file and RTSP sources; the output sink is selected by configuration or the
environment overrides above.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

CONFIG_PATH="$1"
OUTPUT_DIR="${2:-}"
OUTPUT_SINK="${OUTPUT_SINK:-}"
OUTPUT_URL="${OUTPUT_URL:-}"
OUTPUT_VIDEO_PATH="${OUTPUT_VIDEO_PATH:-}"
RUNTIME_DIR="${RUNTIME_DIR:-}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-}"
ALL_CLASSES="${ALL_CLASSES:-0}"
RUN_SECONDS="${RUN_SECONDS:-0}"
START_UI="${START_UI:-0}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

args=(--config "$CONFIG_PATH" --run-seconds "$RUN_SECONDS")
if [ -n "$OUTPUT_DIR" ]; then
    args+=(--output-dir "$OUTPUT_DIR")
fi
if [ -n "$OUTPUT_SINK" ]; then
    args+=(--output-sink "$OUTPUT_SINK")
fi
if [ -n "$OUTPUT_URL" ]; then
    args+=(--output-url "$OUTPUT_URL")
fi
if [ -n "$OUTPUT_VIDEO_PATH" ]; then
    args+=(--output-video "$OUTPUT_VIDEO_PATH")
fi
if [ -n "$RUNTIME_DIR" ]; then
    args+=(--runtime-dir "$RUNTIME_DIR")
fi
if [ -n "$CONFIDENCE_THRESHOLD" ]; then
    args+=(--confidence-threshold "$CONFIDENCE_THRESHOLD")
fi
if [ "$ALL_CLASSES" = "1" ]; then
    args+=(--all-classes)
fi
if [ "$START_UI" != "1" ]; then
    args+=(--no-web)
fi

source scripts/deploy/env.sh
PYTHONPATH=src python3 -m app.main "${args[@]}"
