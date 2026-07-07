#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_VIDEO="/home/nvidia/Desktop/YOLO/video/1.mp4"
INPUT_VIDEO="${1:-$DEFAULT_VIDEO}"
OUTPUT_VIDEO="${2:-outputs/smoke/person_analytics.mp4}"
OUTPUT_JSON="${3:-outputs/smoke/analytics_results.jsonl}"
OUTPUT_SUMMARY="${4:-outputs/smoke/analytics_summary.json}"
OUTPUT_OVERLAY="${5:-outputs/smoke/person_analytics_overlay.mp4}"
ANALYTICS_CONFIG="${ANALYTICS_CONFIG:-configs/analytics/person_analytics.yaml}"
OUTPUT_DIR="$(dirname "$OUTPUT_VIDEO")"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Smoke input video not found: $INPUT_VIDEO" >&2
    exit 1
fi

echo "== Person Analytics Smoke Test =="
echo "Input video: $INPUT_VIDEO"
echo "Analytics config: $ANALYTICS_CONFIG"
echo "Output video: $OUTPUT_VIDEO"
echo "Output JSONL: $OUTPUT_JSON"
echo "Output summary: $OUTPUT_SUMMARY"
echo "Output overlay video: $OUTPUT_OVERLAY"
echo ""

ANALYTICS_CONFIG="$ANALYTICS_CONFIG" \
OUTPUT_VIDEO_NAME="$(basename "$OUTPUT_VIDEO")" \
OUTPUT_JSON_NAME="$(basename "$OUTPUT_JSON")" \
OUTPUT_SUMMARY_NAME="$(basename "$OUTPUT_SUMMARY")" \
OUTPUT_OVERLAY_NAME="$(basename "$OUTPUT_OVERLAY")" \
scripts/run_person_analytics.sh "$INPUT_VIDEO" "$OUTPUT_DIR"

echo ""
echo "Analytics smoke test passed."
