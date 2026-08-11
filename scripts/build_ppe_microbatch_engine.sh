#!/usr/bin/env bash
set -euo pipefail

# Build a dedicated PPE engine for the micro-batch experiment. It never
# overwrites the batch-1 deployment engine.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

source scripts/env.sh >/dev/null

MAX_BATCH=""
ONNX_PATH="models/ppe_yolov8n_dynamic.onnx"
ENGINE_PATH=""
INPUT_NAME="images"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --max-batch) MAX_BATCH="$2"; shift 2 ;;
        --onnx) ONNX_PATH="$2"; shift 2 ;;
        --engine) ENGINE_PATH="$2"; shift 2 ;;
        --input-name) INPUT_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --max-batch {4|8} [--onnx PATH] [--engine PATH] [--input-name images]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$MAX_BATCH" in
    4|8) ;;
    *) echo "--max-batch must be 4 or 8" >&2; exit 2 ;;
esac

TRTEXEC="${TRTEXEC:-$(command -v trtexec || true)}"
if [ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
    echo "trtexec not found; run source scripts/env.sh first." >&2
    exit 1
fi
if [ ! -s "$ONNX_PATH" ]; then
    echo "PPE ONNX model not found: $ONNX_PATH" >&2
    exit 1
fi

ENGINE_PATH="${ENGINE_PATH:-models/fp16/ppe_yolov8n_dynamic_fp16_b${MAX_BATCH}.engine}"
TEMP_ENGINE="${ENGINE_PATH}.tmp"
mkdir -p "$(dirname "$ENGINE_PATH")"
rm -f "$TEMP_ENGINE"
trap 'rm -f "$TEMP_ENGINE"' EXIT

"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --saveEngine="$TEMP_ENGINE" \
    --fp16 \
    --skipInference \
    --memPoolSize=workspace:2048M \
    --minShapes="${INPUT_NAME}:1x3x640x640" \
    --optShapes="${INPUT_NAME}:${MAX_BATCH}x3x640x640" \
    --maxShapes="${INPUT_NAME}:${MAX_BATCH}x3x640x640"

test -s "$TEMP_ENGINE"
mv -f "$TEMP_ENGINE" "$ENGINE_PATH"
trap - EXIT
echo "Built PPE micro-batch engine: $ENGINE_PATH"
