#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/project_paths.sh"

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_rtsp_individual_outputs.sh [INPUT_VIDEO_DIR] [OUTPUT_DIR]

Purpose:
  Start N simulated RTSP streams, then write one independent OSD MP4 per stream.
  This avoids the 2x4 tiler/merged-video path.

Defaults:
  INPUT_VIDEO_DIR=$VIDEO_DIR
  OUTPUT_DIR=$OUTPUT_ROOT/rtsp_individual_outputs

Environment:
  SOURCE_COUNT=8
  INDIVIDUAL_JOBS=1
  RTSP_PORT=8557
  RTSP_HOST=127.0.0.1
  MOUNT_PREFIX=stream
  RUN_SECONDS=40
  OUTPUT_WIDTH=640
  OUTPUT_HEIGHT=640
  ENCODER_BITRATE=8000000
  CONFIDENCE_THRESHOLD=0.25
  SIM_TRANSCODE=0

Examples:
  scripts/run_rtsp_individual_outputs.sh "$VIDEO_DIR"

  SOURCE_COUNT=8 INDIVIDUAL_JOBS=2 RUN_SECONDS=40 \
    scripts/run_rtsp_individual_outputs.sh "$VIDEO_DIR" outputs/rtsp_individual_8
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

INPUT_DIR="${1:-$VIDEO_DIR}"
OUTPUT_DIR="${2:-$OUTPUT_ROOT/rtsp_individual_outputs}"
SOURCE_COUNT="${SOURCE_COUNT:-8}"
INDIVIDUAL_JOBS="${INDIVIDUAL_JOBS:-1}"
RTSP_PORT="${RTSP_PORT:-8557}"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
MOUNT_PREFIX="${MOUNT_PREFIX:-stream}"
RUN_SECONDS="${RUN_SECONDS:-40}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-640}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-640}"
ENCODER_BITRATE="${ENCODER_BITRATE:-8000000}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"

if ! [[ "$SOURCE_COUNT" =~ ^[0-9]+$ ]] || [ "$SOURCE_COUNT" -lt 1 ]; then
    echo "SOURCE_COUNT must be a positive integer: $SOURCE_COUNT" >&2
    exit 1
fi
if ! [[ "$INDIVIDUAL_JOBS" =~ ^[0-9]+$ ]] || [ "$INDIVIDUAL_JOBS" -lt 1 ]; then
    echo "INDIVIDUAL_JOBS must be a positive integer: $INDIVIDUAL_JOBS" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "== RTSP Individual OSD Outputs =="
echo "Input video directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Source count: $SOURCE_COUNT"
echo "Parallel individual jobs: $INDIVIDUAL_JOBS"
echo "RTSP base: rtsp://$RTSP_HOST:$RTSP_PORT/$MOUNT_PREFIX"
echo "Run seconds: $RUN_SECONDS"
echo "Output size: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "Encoder bitrate: $ENCODER_BITRATE"
echo "Simulator transcode: ${SIM_TRANSCODE:-0}"
echo ""

RTSP_PORT="$RTSP_PORT" \
RTSP_HOST="$RTSP_HOST" \
MOUNT_PREFIX="$MOUNT_PREFIX" \
RUNTIME_DIR=".runtime/mediamtx_individual" \
scripts/simulate_cameras.sh "$INPUT_DIR" --limit "$SOURCE_COUNT"

sleep 3

pids=()
failed=0

run_one() {
    local index="$1"
    local stream_id
    local stream_uri
    local stream_dir
    stream_id="$(printf 'stream_%02d' "$index")"
    stream_uri="rtsp://$RTSP_HOST:$RTSP_PORT/${MOUNT_PREFIX}${index}"
    stream_dir="$OUTPUT_DIR/$stream_id"
    mkdir -p "$stream_dir"

    echo "[$index/$SOURCE_COUNT] $stream_uri -> $stream_dir/${stream_id}_osd.mp4"
    SOURCE_COUNT=1 \
    RTSP_URIS="$stream_uri" \
    OUTPUT_SINK=file \
    OUTPUT_WIDTH="$OUTPUT_WIDTH" \
    OUTPUT_HEIGHT="$OUTPUT_HEIGHT" \
    ENABLE_TILER=0 \
    RUN_SECONDS="$RUN_SECONDS" \
    CONFIDENCE_THRESHOLD="$CONFIDENCE_THRESHOLD" \
    SOURCE_STATUS_PATH=".runtime/mediamtx_individual/source_status.json" \
    ENCODER_BITRATE="$ENCODER_BITRATE" \
    scripts/run_rtsp_inproc.sh "$stream_dir"

    if [ -f "$stream_dir/rtsp_preview.mp4" ]; then
        mv "$stream_dir/rtsp_preview.mp4" "$stream_dir/${stream_id}_osd.mp4"
    fi
}

wait_for_one_batch() {
    local pid
    local status
    for pid in "${pids[@]}"; do
        status=0
        wait "$pid" || status=$?
        if [ "$status" -ne 0 ]; then
            failed=1
        fi
    done
    pids=()
}

index=1
while [ "$index" -le "$SOURCE_COUNT" ]; do
    run_one "$index" &
    pids+=("$!")
    if [ "${#pids[@]}" -ge "$INDIVIDUAL_JOBS" ]; then
        wait_for_one_batch
    fi
    index=$((index + 1))
done

if [ "${#pids[@]}" -gt 0 ]; then
    wait_for_one_batch
fi

python3 - "$OUTPUT_DIR/individual_outputs.json" "$SOURCE_COUNT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).parent
source_count = int(sys.argv[2])
items = []
for index in range(1, source_count + 1):
    stream_id = f"stream_{index:02d}"
    stream_dir = root / stream_id
    video = stream_dir / f"{stream_id}_osd.mp4"
    jsonl = stream_dir / "results.jsonl"
    run_log = stream_dir / "run.log"
    items.append(
        {
            "stream_id": stream_id,
            "video": str(video),
            "video_exists": video.is_file() and video.stat().st_size > 0,
            "jsonl": str(jsonl),
            "jsonl_exists": jsonl.is_file() and jsonl.stat().st_size > 0,
            "run_log": str(run_log),
        }
    )
payload = {
    "mode": "rtsp_individual_outputs",
    "source_count": source_count,
    "outputs": items,
    "failed_count": len([item for item in items if not item["video_exists"]]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote individual output index: {sys.argv[1]}")
PY

echo ""
echo "Individual RTSP outputs completed."
echo "Index: $OUTPUT_DIR/individual_outputs.json"
echo "Videos:"
find "$OUTPUT_DIR" -maxdepth 2 -name '*_osd.mp4' -print | sort
echo ""
echo "Check packet drops:"
echo "grep -E \"reader is too slow|discarding|RTP packets are too big|ERROR|Traceback\" .runtime/mediamtx_individual/mediamtx.log | tail -80"

exit "$failed"
