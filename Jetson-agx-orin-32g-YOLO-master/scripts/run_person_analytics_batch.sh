#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_person_analytics_batch.sh INPUT_VIDEO_DIR [OUTPUT_BATCH_DIR]

Optional environment overrides:
  ANALYTICS_CONFIG=configs/analytics/person_analytics.yaml
  VIDEO_GLOB=*.mp4
  CONTINUE_ON_ERROR=1
  SKIP_OVERLAY=0
  SKIP_CHECK=0
  OUTPUT_WIDTH=1280
  OUTPUT_HEIGHT=720
  CONFIDENCE_THRESHOLD=0.25

Example:
  scripts/run_person_analytics_batch.sh \
    /home/nvidia/Desktop/YOLO/video \
    outputs/batch
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

INPUT_DIR="$1"
BATCH_DIR="${2:-outputs/batch}"
ANALYTICS_CONFIG="${ANALYTICS_CONFIG:-configs/analytics/person_analytics.yaml}"
VIDEO_GLOB="${VIDEO_GLOB:-*.mp4}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Input video directory not found: $INPUT_DIR" >&2
    exit 1
fi
if [ ! -f "$ANALYTICS_CONFIG" ]; then
    echo "Analytics config not found: $ANALYTICS_CONFIG" >&2
    exit 1
fi

mkdir -p "$BATCH_DIR"

mapfile -t VIDEOS < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "$VIDEO_GLOB" | sort)
if [ "${#VIDEOS[@]}" -eq 0 ]; then
    echo "No videos matched $VIDEO_GLOB in $INPUT_DIR" >&2
    exit 1
fi

echo "== Person Analytics Batch Run =="
echo "Input directory: $INPUT_DIR"
echo "Video glob: $VIDEO_GLOB"
echo "Video count: ${#VIDEOS[@]}"
echo "Output batch directory: $BATCH_DIR"
echo "Analytics config: $ANALYTICS_CONFIG"
echo ""

batch_failures=0
index=0
for video in "${VIDEOS[@]}"; do
    index=$((index + 1))
    stem="$(basename "$video")"
    stem="${stem%.*}"
    safe_stem="$(printf '%s' "$stem" | tr -c '[:alnum:]_.-' '_')"
    output_dir="$BATCH_DIR/$(printf '%03d_%s' "$index" "$safe_stem")"
    metadata_path="$output_dir/run_metadata.json"
    mkdir -p "$output_dir"

    started_at="$(date --iso-8601=seconds)"
    echo "[$index/${#VIDEOS[@]}] Processing $video"

    ANALYTICS_CONFIG="$ANALYTICS_CONFIG" scripts/run_person_analytics.sh "$video" "$output_dir"
    exit_code=$?
    finished_at="$(date --iso-8601=seconds)"

    if [ "$exit_code" -eq 0 ]; then
        status="ok"
        error=""
        echo "[$index/${#VIDEOS[@]}] OK"
    else
        status="failed"
        error="run_person_analytics.sh exited with $exit_code"
        batch_failures=$((batch_failures + 1))
        echo "[$index/${#VIDEOS[@]}] FAIL: $error" >&2
    fi

    python3 - "$metadata_path" "$video" "$status" "$exit_code" "$started_at" "$finished_at" "$error" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "input_video": sys.argv[2],
    "status": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "started_at": sys.argv[5],
    "finished_at": sys.argv[6],
    "error": sys.argv[7],
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    if [ "$exit_code" -ne 0 ] && [ "$CONTINUE_ON_ERROR" != "1" ]; then
        break
    fi
done

python3 scripts/summarize_person_batch.py "$BATCH_DIR" "$BATCH_DIR/batch_summary.json"
python3 scripts/export_person_batch_report.py "$BATCH_DIR/batch_summary.json" "$BATCH_DIR"
python3 scripts/check_person_batch_outputs.py "$BATCH_DIR/batch_summary.json" "$BATCH_DIR/batch_quality.json" || true

echo ""
echo "Batch completed with failures: $batch_failures"
echo "Batch summary: $BATCH_DIR/batch_summary.json"
echo "Batch CSV: $BATCH_DIR/batch_summary.csv"
echo "Batch HTML report: $BATCH_DIR/batch_report.html"
echo "Batch quality: $BATCH_DIR/batch_quality.json"
exit "$batch_failures"
