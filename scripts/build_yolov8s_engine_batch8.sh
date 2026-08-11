#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ONNX_PATH="${ONNX_PATH:-models/yolov8s.onnx}"
ENGINE_PATH="${ENGINE_PATH:-models/yolov8s.engine}"
INPUT_NAME="${INPUT_NAME:-input}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
TRTEXEC="${TRTEXEC:-}"

if [ -z "$TRTEXEC" ]; then
    TRTEXEC="$(command -v trtexec || true)"
fi
if [ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
    echo "trtexec not found. Install the Jetson TensorRT command-line tools first." >&2
    exit 1
fi
if [ ! -s "$ONNX_PATH" ]; then
    echo "ONNX model not found or empty: $ONNX_PATH" >&2
    exit 1
fi

TEMP_ENGINE="${ENGINE_PATH}.batch8.tmp"
BACKUP_ENGINE="${ENGINE_PATH}.before_batch8"
trap 'rm -f "$TEMP_ENGINE"' EXIT

echo "Building TensorRT engine for dynamic batch 1..8"
echo "  trtexec: $TRTEXEC"
echo "  ONNX: $ONNX_PATH"
echo "  output: $ENGINE_PATH"
echo "  input: $INPUT_NAME"

"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --saveEngine="$TEMP_ENGINE" \
    --fp16 \
    --memPoolSize="workspace:${WORKSPACE_MB}M" \
    --minShapes="${INPUT_NAME}:1x3x640x640" \
    --optShapes="${INPUT_NAME}:8x3x640x640" \
    --maxShapes="${INPUT_NAME}:8x3x640x640"

if [ ! -s "$TEMP_ENGINE" ]; then
    echo "TensorRT did not produce an engine: $TEMP_ENGINE" >&2
    exit 1
fi

if [ -f "$ENGINE_PATH" ]; then
    cp -f "$ENGINE_PATH" "$BACKUP_ENGINE"
    echo "Backed up current engine to $BACKUP_ENGINE"
fi
mv -f "$TEMP_ENGINE" "$ENGINE_PATH"
trap - EXIT

echo "Batch-8 engine installed: $ENGINE_PATH"
ls -lh "$ENGINE_PATH"
