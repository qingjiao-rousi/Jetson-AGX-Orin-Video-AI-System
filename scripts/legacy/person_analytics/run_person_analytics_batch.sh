#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR" || exit 1

usage() {
    cat <<'USAGE'
Usage:
  scripts/legacy/person_analytics/run_person_analytics_batch.sh INPUT_VIDEO_DIR [OUTPUT_BATCH_DIR]

Optional environment overrides:
  ANALYTICS_CONFIG=configs/legacy/person_analytics.yaml
  VIDEO_GLOB=*.mp4
  BATCH_JOBS=8
  CONTINUE_ON_ERROR=1
  SKIP_OVERLAY=0
  SKIP_CHECK=0
  OUTPUT_WIDTH=1280
  OUTPUT_HEIGHT=720
  CONFIDENCE_THRESHOLD=0.25

Examples:
  scripts/legacy/person_analytics/run_person_analytics_batch.sh "$VIDEO_DIR" "$OUTPUT_ROOT/batch"

  BATCH_JOBS=4 scripts/legacy/person_analytics/run_person_analytics_batch.sh "$VIDEO_DIR" "$OUTPUT_ROOT/batch"
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

INPUT_DIR="$1"
BATCH_DIR="${2:-outputs/batch}"
ANALYTICS_CONFIG="${ANALYTICS_CONFIG:-configs/legacy/person_analytics.yaml}"
VIDEO_GLOB="${VIDEO_GLOB:-*.mp4}"
BATCH_JOBS="${BATCH_JOBS:-8}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Input video directory not found: $INPUT_DIR" >&2
    exit 1
fi
if [ ! -f "$ANALYTICS_CONFIG" ]; then
    echo "Analytics config not found: $ANALYTICS_CONFIG" >&2
    exit 1
fi
if ! [[ "$BATCH_JOBS" =~ ^[0-9]+$ ]] || [ "$BATCH_JOBS" -lt 1 ]; then
    echo "BATCH_JOBS must be a positive integer: $BATCH_JOBS" >&2
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
echo "Parallel jobs: $BATCH_JOBS"
echo ""

run_one_video() {
    local index="$1"
    local total="$2"
    local video="$3"
    local output_dir="$4"
    local metadata_path="$output_dir/run_metadata.json"
    local log_path="$output_dir/run.log"
    local runtime_dir="$output_dir/.runtime"
    local started_at
    local finished_at
    local status
    local error
    local exit_code

    mkdir -p "$output_dir" "$runtime_dir"
    started_at="$(date --iso-8601=seconds)"

    {
        echo "[$index/$total] Processing $video"
        echo "Output directory: $output_dir"
        echo "Runtime directory: $runtime_dir"
        echo "Started at: $started_at"
        ANALYTICS_CONFIG="$ANALYTICS_CONFIG" RUNTIME_DIR="$runtime_dir" \
            scripts/legacy/person_analytics/run_person_analytics.sh "$video" "$output_dir"
    } >"$log_path" 2>&1
    exit_code=$?
    finished_at="$(date --iso-8601=seconds)"

    if [ "$exit_code" -eq 0 ]; then
        status="ok"
        error=""
    else
        status="failed"
        error="run_person_analytics.sh exited with $exit_code"
    fi

    python3 - "$metadata_path" "$video" "$status" "$exit_code" "$started_at" "$finished_at" "$error" "$log_path" "$BATCH_JOBS" <<'PY'
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
    "log_path": sys.argv[8],
    "batch_jobs": int(sys.argv[9]),
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    if [ "$exit_code" -eq 0 ]; then
        echo "[$index/$total] OK: $video"
    else
        echo "[$index/$total] FAIL: $video ($error). See $log_path" >&2
    fi
    return "$exit_code"
}

wait_for_available_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$BATCH_JOBS" ]; do
        sleep 0.5
    done
}

batch_failures=0
launched=0
stop_launching=0
declare -a PIDS=()
declare -A PID_LABELS=()

for video in "${VIDEOS[@]}"; do
    if [ "$stop_launching" -eq 1 ]; then
        break
    fi

    wait_for_available_slot

    launched=$((launched + 1))
    stem="$(basename "$video")"
    stem="${stem%.*}"
    safe_stem="$(printf '%s' "$stem" | tr -c '[:alnum:]_.-' '_')"
    output_dir="$BATCH_DIR/$(printf '%03d_%s' "$launched" "$safe_stem")"

    echo "[$launched/${#VIDEOS[@]}] Launching $video -> $output_dir"
    run_one_video "$launched" "${#VIDEOS[@]}" "$video" "$output_dir" &
    pid=$!
    PIDS+=("$pid")
    PID_LABELS["$pid"]="[$launched/${#VIDEOS[@]}] $video"

    if [ "$CONTINUE_ON_ERROR" != "1" ]; then
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                if ! wait "$pid"; then
                    batch_failures=$((batch_failures + 1))
                    stop_launching=1
                fi
            fi
        done
    fi
done

for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        :
    else
        batch_failures=$((batch_failures + 1))
        echo "Job failed: ${PID_LABELS[$pid]}" >&2
    fi
done

python3 scripts/legacy/person_analytics/summarize_person_batch.py "$BATCH_DIR" "$BATCH_DIR/batch_summary.json"
python3 scripts/legacy/person_analytics/export_person_batch_report.py "$BATCH_DIR/batch_summary.json" "$BATCH_DIR"
python3 scripts/legacy/person_analytics/check_person_batch_outputs.py "$BATCH_DIR/batch_summary.json" "$BATCH_DIR/batch_quality.json" || true

echo ""
echo "Batch completed with failures: $batch_failures"
echo "Batch summary: $BATCH_DIR/batch_summary.json"
echo "Batch CSV: $BATCH_DIR/batch_summary.csv"
echo "Batch HTML report: $BATCH_DIR/batch_report.html"
echo "Batch quality: $BATCH_DIR/batch_quality.json"
exit "$batch_failures"
