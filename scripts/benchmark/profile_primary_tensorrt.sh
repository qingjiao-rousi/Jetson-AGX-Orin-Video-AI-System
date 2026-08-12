#!/usr/bin/env bash
set -euo pipefail

# Export repeatable per-layer TensorRT profiles for the primary detector only.
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

source scripts/deploy/env.sh >/dev/null

BATCH_SIZE=8
DURATION_SECONDS=30
OUTPUT_DIR=""
FP16_ENGINE="models/fp16/yolov8s.engine"
INT8_ENGINE="models/int8/yolov8s_coco_train504.engine"

usage() {
    cat <<'USAGE'
Usage: scripts/benchmark/profile_primary_tensorrt.sh [options]

Options:
  --batch-size N       TensorRT input batch to profile (default: 8)
  --duration SECONDS   Measured duration after warmup (default: 30)
  --output-dir PATH    Output directory (default: outputs/profiling/tensorrt/<UTC>)
  --fp16-engine PATH   FP16 primary engine path
  --int8-engine PATH   INT8 primary engine path

The script measures standalone TensorRT engines. It is not a DeepStream
end-to-end FPS measurement and does not include decode, tracker, Python tasks,
or output handling.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --duration) DURATION_SECONDS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --fp16-engine) FP16_ENGINE="$2"; shift 2 ;;
        --int8-engine) INT8_ENGINE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "--batch-size must be a positive integer" >&2
    exit 2
fi
if ! [[ "$DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--duration must be a positive integer" >&2
    exit 2
fi

TRTEXEC="${TRTEXEC:-$(command -v trtexec || true)}"
if [ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
    echo "trtexec not found; install libnvinfer-bin or source the Jetson environment." >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/profiling/tensorrt/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUTPUT_DIR"

profile_engine() {
    local name="$1"
    local engine="$2"
    local prefix="$OUTPUT_DIR/${name}_b${BATCH_SIZE}"
    if [ ! -s "$engine" ]; then
        echo "Engine not found: $engine" >&2
        return 1
    fi
    echo "Profiling $name: $engine"
    "$TRTEXEC" \
        --loadEngine="$engine" \
        --shapes="input:${BATCH_SIZE}x3x640x640" \
        --warmUp=200 \
        --duration="$DURATION_SECONDS" \
        --useSpinWait \
        --profilingVerbosity=detailed \
        --dumpProfile \
        --exportProfile="${prefix}.profile.json" \
        --exportTimes="${prefix}.times.json" \
        2>&1 | tee "${prefix}.log"
}

profile_engine fp16 "$FP16_ENGINE"
profile_engine int8 "$INT8_ENGINE"
printf 'TensorRT profiles written to: %s\n' "$OUTPUT_DIR"
