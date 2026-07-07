#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_VIDEO="/home/nvidia/Desktop/YOLO/video/1.mp4"
INPUT_VIDEO="${1:-$DEFAULT_VIDEO}"
OUTPUT_VIDEO="${2:-outputs/smoke/person_counting.mp4}"
OUTPUT_JSON="${3:-outputs/smoke/counting_results.jsonl}"
OUTPUT_SUMMARY="${4:-outputs/smoke/summary.json}"
MIN_TRACK_FRAMES="${MIN_TRACK_FRAMES:-2}"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Smoke input video not found: $INPUT_VIDEO" >&2
    exit 1
fi

echo "== Person Counting Smoke Test =="
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Output JSONL: $OUTPUT_JSON"
echo "Output summary: $OUTPUT_SUMMARY"
echo "Minimum track frames: $MIN_TRACK_FRAMES"
echo ""

scripts/run_person_detect.sh "$INPUT_VIDEO" "$OUTPUT_VIDEO" "$OUTPUT_JSON"

echo ""
REQUIRE_TRACKS=1 scripts/check_person_output.sh "$OUTPUT_VIDEO" "$OUTPUT_JSON"

echo ""
python3 scripts/summarize_person_tracks.py \
    "$OUTPUT_JSON" \
    "$OUTPUT_SUMMARY" \
    --min-track-frames "$MIN_TRACK_FRAMES"

echo ""
python3 - "$OUTPUT_SUMMARY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(summary.get("total_unique_persons", 0)) <= 0:
    print("[FAIL] total_unique_persons is zero")
    raise SystemExit(1)
if int(summary.get("frames_with_person", 0)) <= 0:
    print("[FAIL] frames_with_person is zero")
    raise SystemExit(1)
print("[OK] Counting summary is valid")
PY

echo ""
echo "Counting smoke test passed."
