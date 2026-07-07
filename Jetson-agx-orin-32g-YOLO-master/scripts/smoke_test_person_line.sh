#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_VIDEO="/home/nvidia/Desktop/YOLO/video/1.mp4"
INPUT_VIDEO="${1:-$DEFAULT_VIDEO}"
OUTPUT_VIDEO="${2:-outputs/smoke/person_line.mp4}"
OUTPUT_JSON="${3:-outputs/smoke/line_results.jsonl}"
OUTPUT_SUMMARY="${4:-outputs/smoke/line_summary.json}"
LINE="${LINE:-640,0,640,720}"
LINE_ID="${LINE_ID:-middle-vertical}"
MIN_SIDE_DISTANCE="${MIN_SIDE_DISTANCE:-1.0}"
COUNT_ONCE_PER_TRACK="${COUNT_ONCE_PER_TRACK:-1}"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Smoke input video not found: $INPUT_VIDEO" >&2
    exit 1
fi

echo "== Person Line Crossing Smoke Test =="
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Output JSONL: $OUTPUT_JSON"
echo "Output line summary: $OUTPUT_SUMMARY"
echo "Line: $LINE_ID=$LINE"
echo "Minimum side distance: $MIN_SIDE_DISTANCE"
echo "Count once per track: $COUNT_ONCE_PER_TRACK"
echo ""

scripts/run_person_detect.sh "$INPUT_VIDEO" "$OUTPUT_VIDEO" "$OUTPUT_JSON"

echo ""
REQUIRE_TRACKS=1 scripts/check_person_output.sh "$OUTPUT_VIDEO" "$OUTPUT_JSON"

echo ""
LINE_ARGS=(
    "$OUTPUT_JSON" \
    "$OUTPUT_SUMMARY" \
    --line "$LINE" \
    --line-id "$LINE_ID" \
    --min-side-distance "$MIN_SIDE_DISTANCE"
)

if [ "$COUNT_ONCE_PER_TRACK" = "1" ]; then
    LINE_ARGS+=(--count-once-per-track)
else
    LINE_ARGS+=(--no-count-once-per-track)
fi

python3 scripts/summarize_person_line.py "${LINE_ARGS[@]}"

echo ""
python3 - "$OUTPUT_SUMMARY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(summary.get("total_frames", 0)) <= 0:
    print("[FAIL] total_frames is zero")
    raise SystemExit(1)
print("[OK] Line crossing summary is valid")
PY

echo ""
echo "Line crossing smoke test passed."
