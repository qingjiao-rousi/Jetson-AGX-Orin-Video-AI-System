#!/usr/bin/env bash
set -euo pipefail

# Capture one bounded DeepStream pipeline run with Nsight Systems.
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="configs/app/app_multifile_8_primary_int8_isolated_tasks.yaml"
OUTPUT_DIR=""
RUN_SECONDS=60
SINK=fake

usage() {
    cat <<'USAGE'
Usage: scripts/benchmark/profile_pipeline_nsys.sh [options]

Options:
  --config PATH         Application YAML to profile
  --output-dir PATH     Output directory (default: outputs/profiling/nsys/<UTC>)
  --run-seconds N       Bounded application duration (default: 60)
  --sink fake|file      Output sink during capture (default: fake)

The resulting .nsys-rep and CSV summaries are local artifacts. Profiling adds
overhead, so do not use this run as a directly comparable benchmark datapoint.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --run-seconds) RUN_SECONDS="$2"; shift 2 ;;
        --sink) SINK="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$RUN_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--run-seconds must be a positive integer" >&2
    exit 2
fi
case "$SINK" in fake|file) ;; *) echo "--sink must be fake or file" >&2; exit 2 ;; esac
if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found. Install Nsight Systems CLI for this JetPack release and add it to PATH." >&2
    exit 1
fi
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/profiling/nsys/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUTPUT_DIR"
REPORT="$OUTPUT_DIR/pipeline"
APP_OUTPUT="$OUTPUT_DIR/application"

nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cpuctxsw=none \
    --output="$REPORT" \
    env OUTPUT_SINK="$SINK" RUN_SECONDS="$RUN_SECONDS" ENABLE_TEGRASTATS=1 \
    scripts/deploy/run_multistream.sh "$CONFIG_PATH" "$APP_OUTPUT"

nsys stats \
    --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum \
    --format csv \
    --output "$OUTPUT_DIR/nsys_stats" \
    "${REPORT}.nsys-rep"

printf 'Nsight Systems report: %s.nsys-rep\n' "$REPORT"
printf 'Nsight Systems summaries: %s\n' "$OUTPUT_DIR"
