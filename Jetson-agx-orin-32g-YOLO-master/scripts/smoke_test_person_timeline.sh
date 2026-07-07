#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_VIDEO="/home/nvidia/Desktop/YOLO/video/1.mp4"
INPUT_VIDEO="${1:-$DEFAULT_VIDEO}"
OUTPUT_VIDEO="${2:-outputs/smoke/person_timeline.mp4}"
OUTPUT_JSON="${3:-outputs/smoke/timeline_results.jsonl}"
OUTPUT_TIMELINE="${4:-outputs/smoke/timeline_summary.json}"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Smoke input video not found: $INPUT_VIDEO" >&2
    exit 1
fi

echo "== Person Timeline Smoke Test =="
echo "Input video: $INPUT_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Output JSONL: $OUTPUT_JSON"
echo "Output timeline summary: $OUTPUT_TIMELINE"
echo ""

scripts/run_person_detect.sh "$INPUT_VIDEO" "$OUTPUT_VIDEO" "$OUTPUT_JSON"

echo ""
scripts/check_person_output.sh "$OUTPUT_VIDEO" "$OUTPUT_JSON"

echo ""
python3 scripts/summarize_person_timeline.py "$OUTPUT_JSON" "$OUTPUT_TIMELINE"

echo ""
python3 - "$OUTPUT_TIMELINE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(summary.get("stream_count", 0)) <= 0:
    print("[FAIL] stream_count is zero")
    raise SystemExit(1)
for stream in summary["streams"].values():
    if int(stream.get("frame_count", 0)) <= 0:
        print("[FAIL] stream frame_count is zero")
        raise SystemExit(1)
    if not stream.get("is_frame_continuous"):
        print("[FAIL] stream frames are not continuous")
        raise SystemExit(1)
print("[OK] Timeline summary is valid")
PY

echo ""
echo "Timeline smoke test passed."
