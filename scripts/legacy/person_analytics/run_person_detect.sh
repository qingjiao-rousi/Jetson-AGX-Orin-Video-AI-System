#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/deploy/project_paths.sh"

usage() {
    cat <<'USAGE'
Usage:
  scripts/legacy/person_analytics/run_person_detect.sh INPUT_MP4 [OUTPUT_MP4] [OUTPUT_JSONL]

Environment overrides:
  OUTPUT_WIDTH=1280
  OUTPUT_HEIGHT=720
  CONFIDENCE_THRESHOLD=0.25
  RUNTIME_DIR=outputs/runtime

Example:
  scripts/legacy/person_analytics/run_person_detect.sh \
    "$VIDEO_DIR/input.mp4" \
    outputs/person_detect.mp4 \
    outputs/results.jsonl
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

INPUT_VIDEO="$1"
OUTPUT_VIDEO="${2:-outputs/person_detect.mp4}"
OUTPUT_JSON="${3:-outputs/results.jsonl}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-1280}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-720}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"
ENABLE_WEB="${ENABLE_WEB:-0}"
RUNTIME_DIR="${RUNTIME_DIR:-outputs/runtime}"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Input video not found: $INPUT_VIDEO" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_VIDEO")" "$(dirname "$OUTPUT_JSON")" "$RUNTIME_DIR"
rm -f "$OUTPUT_VIDEO" "$OUTPUT_JSON"

# shellcheck disable=SC1091
source scripts/deploy/env.sh

APP_ARGS=(
    --config configs/app/app_minimal.yaml \
    --input-video "$INPUT_VIDEO" \
    --output-video "$OUTPUT_VIDEO" \
    --output-json "$OUTPUT_JSON" \
    --output-width "$OUTPUT_WIDTH" \
    --output-height "$OUTPUT_HEIGHT" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --runtime-dir "$RUNTIME_DIR"
)

if [ "$ENABLE_WEB" != "1" ]; then
    APP_ARGS+=(--no-web)
fi

PYTHONPATH=src python3 -m app.main "${APP_ARGS[@]}"

echo ""
echo "Wrote video: $OUTPUT_VIDEO"
echo "Wrote JSONL: $OUTPUT_JSON"
