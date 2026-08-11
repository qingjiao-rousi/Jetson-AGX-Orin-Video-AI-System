#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/deploy/project_paths.sh"

RETENTION_DAYS="${RETENTION_DAYS:-7}"
MAX_OUTPUT_GB="${MAX_OUTPUT_GB:-20}"
DRY_RUN="${DRY_RUN:-0}"

delete_path() {
    local path="$1"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] delete $path"
    else
        rm -rf -- "$path"
        echo "deleted $path"
    fi
}

echo "== Cleanup Outputs =="
echo "Output root: $OUTPUT_ROOT"
echo "Runtime root: $RUNTIME_ROOT"
echo "Retention days: $RETENTION_DAYS"
echo "Max output GB: $MAX_OUTPUT_GB"
echo "Dry run: $DRY_RUN"

mkdir -p "$OUTPUT_ROOT" "$RUNTIME_ROOT"

while IFS= read -r path; do
    delete_path "$path"
done < <(
    find "$OUTPUT_ROOT" "$RUNTIME_ROOT" \
        -mindepth 1 \
        \( -name '*.mp4' -o -name '*.jsonl' -o -name '*.log' -o -name '*.old' -o -name '*.tmp' -o -name 'rtsp_acceptance_*' -o -name 'batch_*' -o -name 'bench_*' \) \
        -mtime +"$RETENTION_DAYS" \
        -print 2>/dev/null
)

current_kb="$(du -sk "$OUTPUT_ROOT" 2>/dev/null | awk '{print $1}')"
max_kb=$((MAX_OUTPUT_GB * 1024 * 1024))

if [ "${current_kb:-0}" -le "$max_kb" ]; then
    echo "Output size OK: ${current_kb:-0}KB <= ${max_kb}KB"
    exit 0
fi

echo "Output size exceeds limit; deleting oldest generated files until under limit."
while [ "${current_kb:-0}" -gt "$max_kb" ]; do
    oldest="$(
        find "$OUTPUT_ROOT" -type f \
            \( -name '*.mp4' -o -name '*.jsonl' -o -name '*.log' -o -name '*.json' -o -name '*.csv' -o -name '*.html' \) \
            -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-
    )"
    if [ -z "$oldest" ]; then
        echo "No removable generated files found."
        break
    fi
    delete_path "$oldest"
    current_kb="$(du -sk "$OUTPUT_ROOT" 2>/dev/null | awk '{print $1}')"
done

echo "Cleanup complete. Output size: ${current_kb:-0}KB"
