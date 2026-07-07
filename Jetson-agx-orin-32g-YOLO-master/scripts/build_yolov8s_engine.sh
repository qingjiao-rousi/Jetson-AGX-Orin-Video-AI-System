#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f scripts/env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/env.sh >/dev/null
fi

ONNX_PATH="${ONNX_PATH:-models/yolov8s.onnx}"
ENGINE_PATH="${ENGINE_PATH:-models/yolov8s.engine}"
INPUT_NAME="${INPUT_NAME:-images}"
DYNAMIC_SHAPES="${DYNAMIC_SHAPES:-0}"

if [ ! -f "$ONNX_PATH" ]; then
    echo "Missing ONNX model: $ONNX_PATH"
    exit 1
fi

TRTEXEC="$(command -v trtexec || true)"
if [ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [ -z "$TRTEXEC" ]; then
    echo "trtexec not found. Run: source scripts/env.sh"
    exit 1
fi

TRT_ARGS=(
    --onnx="$ONNX_PATH"
    --saveEngine="$ENGINE_PATH"
    --fp16
)

if [ "$DYNAMIC_SHAPES" = "1" ]; then
    TRT_ARGS+=(
        --minShapes="${INPUT_NAME}:1x3x640x640"
        --optShapes="${INPUT_NAME}:1x3x640x640"
        --maxShapes="${INPUT_NAME}:1x3x640x640"
    )
fi

"$TRTEXEC" "${TRT_ARGS[@]}"

echo "Built engine: $ENGINE_PATH"
