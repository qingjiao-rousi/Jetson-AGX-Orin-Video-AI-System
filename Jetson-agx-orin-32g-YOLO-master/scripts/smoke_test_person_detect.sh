#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_VIDEO="/home/nvidia/Desktop/YOLO/video/2.mp4"
INPUT_VIDEO="${1:-$DEFAULT_VIDEO}"
OUTPUT_VIDEO="${2:-outputs/smoke/2.mp4}"
OUTPUT_JSON="${3:-outputs/smoke/results.jsonl}"

OUTPUT_WIDTH="${OUTPUT_WIDTH:-1280}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-720}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Smoke input video not found: $INPUT_VIDEO" >&2
    echo "Pass a video path explicitly, for example:" >&2
    echo "  scripts/smoke_test_person_detect.sh /home/nvidia/Desktop/YOLO/video/1.mp4" >&2
    exit 1
fi

echo "== Person Detect Smoke Test =="
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Output JSONL: $OUTPUT_JSON"
echo "Output size: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "Confidence threshold: $CONFIDENCE_THRESHOLD"
echo ""

scripts/run_person_detect.sh "$INPUT_VIDEO" "$OUTPUT_VIDEO" "$OUTPUT_JSON"

echo ""
scripts/check_person_output.sh "$OUTPUT_VIDEO" "$OUTPUT_JSON"

echo ""
echo "Smoke test passed."
