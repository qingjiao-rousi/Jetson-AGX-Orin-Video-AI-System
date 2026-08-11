#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR" || exit 1
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/deploy/project_paths.sh"

usage() {
    cat <<'USAGE'
Usage:
  scripts/legacy/person_analytics/run_acceptance_ui.sh [INPUT_VIDEO_DIR] [OUTPUT_BATCH_DIR]

Default:
  INPUT_VIDEO_DIR=$VIDEO_DIR
  OUTPUT_BATCH_DIR=$OUTPUT_ROOT/acceptance_latest

Optional environment overrides:
  BATCH_JOBS=8
  UI_HOST=127.0.0.1
  UI_PORT=8090
  START_UI=1
  ANALYTICS_CONFIG=configs/legacy/person_analytics.yaml
  VIDEO_GLOB=*.mp4
  OUTPUT_WIDTH=1280
  OUTPUT_HEIGHT=720
  CONFIDENCE_THRESHOLD=0.25

Examples:
  scripts/legacy/person_analytics/run_acceptance_ui.sh

  BATCH_JOBS=4 scripts/legacy/person_analytics/run_acceptance_ui.sh \
    "$VIDEO_DIR" \
    "$OUTPUT_ROOT/acceptance_4"

  START_UI=0 scripts/legacy/person_analytics/run_acceptance_ui.sh
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

INPUT_DIR="${1:-$VIDEO_DIR}"
BATCH_DIR="${2:-$OUTPUT_ROOT/acceptance_latest}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8090}"
START_UI="${START_UI:-1}"

echo "== Person Analytics Acceptance Flow =="
echo "Input video directory: $INPUT_DIR"
echo "Output batch directory: $BATCH_DIR"
echo "Parallel jobs: ${BATCH_JOBS:-8}"
echo "UI: http://$UI_HOST:$UI_PORT"
echo ""

scripts/legacy/person_analytics/run_person_analytics_batch.sh "$INPUT_DIR" "$BATCH_DIR"
batch_exit=$?

echo ""
echo "== Acceptance Outputs =="
echo "Batch summary: $BATCH_DIR/batch_summary.json"
echo "Batch quality: $BATCH_DIR/batch_quality.json"
echo "Batch CSV: $BATCH_DIR/batch_summary.csv"
echo "Batch HTML: $BATCH_DIR/batch_report.html"

if [ "$batch_exit" -ne 0 ]; then
    echo "Batch processing finished with failures: $batch_exit" >&2
    echo "UI can still be started to inspect partial results." >&2
fi

if [ "$START_UI" = "1" ]; then
    echo ""
    echo "Starting local dashboard..."
    echo "Open: http://$UI_HOST:$UI_PORT"
    echo "Press Ctrl+C to stop the dashboard."
    python3 scripts/tools/preview_web.py \
        --host "$UI_HOST" \
        --port "$UI_PORT" \
        --batch-dir "$BATCH_DIR"
else
    echo ""
    echo "UI not started because START_UI=$START_UI"
    echo "Start manually with:"
    echo "python3 scripts/tools/preview_web.py --host $UI_HOST --port $UI_PORT --batch-dir $BATCH_DIR"
fi

exit "$batch_exit"
