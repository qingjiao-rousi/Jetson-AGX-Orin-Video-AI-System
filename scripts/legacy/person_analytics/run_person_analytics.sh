#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'USAGE'
Usage:
  scripts/legacy/person_analytics/run_person_analytics.sh INPUT_MP4 [OUTPUT_DIR]

Optional environment overrides:
  ANALYTICS_CONFIG=configs/analytics/person_analytics.yaml
  OUTPUT_VIDEO_NAME=person_analytics.mp4
  OUTPUT_JSON_NAME=results.jsonl
  OUTPUT_SUMMARY_NAME=analytics_summary.json
  OUTPUT_OVERLAY_NAME=person_analytics_overlay.mp4
  OUTPUT_WIDTH=1280
  OUTPUT_HEIGHT=720
  CONFIDENCE_THRESHOLD=0.25
  ENABLE_WEB=0
  SKIP_CHECK=0
  SKIP_OVERLAY=0

Example:
  scripts/legacy/person_analytics/run_person_analytics.sh \
    "$VIDEO_DIR/1.mp4" \
    "$OUTPUT_ROOT/final"
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="${2:-outputs/final}"

ANALYTICS_CONFIG="${ANALYTICS_CONFIG:-configs/analytics/person_analytics.yaml}"
OUTPUT_VIDEO_NAME="${OUTPUT_VIDEO_NAME:-person_analytics.mp4}"
OUTPUT_JSON_NAME="${OUTPUT_JSON_NAME:-results.jsonl}"
OUTPUT_SUMMARY_NAME="${OUTPUT_SUMMARY_NAME:-analytics_summary.json}"
OUTPUT_OVERLAY_NAME="${OUTPUT_OVERLAY_NAME:-person_analytics_overlay.mp4}"
SKIP_CHECK="${SKIP_CHECK:-0}"
SKIP_OVERLAY="${SKIP_OVERLAY:-0}"

OUTPUT_VIDEO="$OUTPUT_DIR/$OUTPUT_VIDEO_NAME"
OUTPUT_JSON="$OUTPUT_DIR/$OUTPUT_JSON_NAME"
OUTPUT_SUMMARY="$OUTPUT_DIR/$OUTPUT_SUMMARY_NAME"
OUTPUT_OVERLAY="$OUTPUT_DIR/$OUTPUT_OVERLAY_NAME"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Input video not found: $INPUT_VIDEO" >&2
    exit 1
fi
if [ ! -f "$ANALYTICS_CONFIG" ]; then
    echo "Analytics config not found: $ANALYTICS_CONFIG" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "== Person Analytics Run =="
echo "Input video: $INPUT_VIDEO"
echo "Analytics config: $ANALYTICS_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Output video: $OUTPUT_VIDEO"
echo "Output JSONL: $OUTPUT_JSON"
echo "Output summary: $OUTPUT_SUMMARY"
echo "Output overlay video: $OUTPUT_OVERLAY"
echo ""

scripts/legacy/person_analytics/run_person_detect.sh "$INPUT_VIDEO" "$OUTPUT_VIDEO" "$OUTPUT_JSON"

if [ "$SKIP_CHECK" != "1" ]; then
    echo ""
    REQUIRE_TRACKS=1 scripts/legacy/person_analytics/check_person_output.sh "$OUTPUT_VIDEO" "$OUTPUT_JSON"
fi

echo ""
python3 scripts/legacy/person_analytics/summarize_person_analytics.py \
    "$OUTPUT_JSON" \
    "$ANALYTICS_CONFIG" \
    "$OUTPUT_SUMMARY"

if [ "$SKIP_OVERLAY" != "1" ]; then
    echo ""
    python3 scripts/legacy/person_analytics/draw_person_analytics.py \
        "$OUTPUT_VIDEO" \
        "$ANALYTICS_CONFIG" \
        "$OUTPUT_OVERLAY"
fi

echo ""
echo "Person analytics completed."
echo "  Video: $OUTPUT_VIDEO"
echo "  JSONL: $OUTPUT_JSON"
echo "  Summary: $OUTPUT_SUMMARY"
if [ "$SKIP_OVERLAY" != "1" ]; then
    echo "  Overlay: $OUTPUT_OVERLAY"
fi
