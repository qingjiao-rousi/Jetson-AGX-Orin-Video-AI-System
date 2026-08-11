#!/usr/bin/env bash
set -euo pipefail

# Rebuild FP16 engines after a TensorRT upgrade. The primary detector supports
# batch 1..8; specialist workers intentionally use their own batch-1 engines.
# TensorRT plans are not forward compatible: do not reuse 10.3 plans on 10.7.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

source scripts/env.sh >/dev/null
TRTEXEC="${TRTEXEC:-$(command -v trtexec || true)}"
if [ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
    echo "trtexec not found; run source scripts/env.sh first." >&2
    exit 1
fi

mkdir -p models/fp16

BUILD_PRIMARY=1
BUILD_SPECIALISTS=1
FORCE_REBUILD="${FORCE_REBUILD:-0}"

for arg in "$@"; do
    case "$arg" in
        --primary-only)
            BUILD_SPECIALISTS=0
            ;;
        --specialists-only)
            BUILD_PRIMARY=0
            ;;
        --force)
            FORCE_REBUILD=1
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: scripts/build_fp16_engines.sh [--primary-only|--specialists-only] [--force]

The primary YOLO engine is built with dynamic batch 1..8 for the DeepStream
streammux benchmark. Specialist workers remain batch 1 and are built once.
Existing engines created by the current TensorRT major.minor version are skipped
unless --force (or FORCE_REBUILD=1) is supplied.
USAGE
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

if [ "$BUILD_PRIMARY" = 0 ] && [ "$BUILD_SPECIALISTS" = 0 ]; then
    echo "Choose either primary or specialist engines." >&2
    exit 2
fi

runtime_series="$(python3 -c 'import tensorrt as trt; print(".".join(trt.__version__.split(".")[:2]))')"

plan_matches_runtime() {
    local engine_path="$1"
    [ -s "$engine_path" ] || return 1
    "$TRTEXEC" --getPlanVersionOnly --loadEngine="$engine_path" 2>&1 \
        | grep -q "Plan was created with TensorRT version ${runtime_series}\."
}

build_engine() {
    local onnx_path="$1"
    local engine_path="$2"
    shift 2
    local temporary="${engine_path}.tmp"

    if [ "$FORCE_REBUILD" != 1 ] && plan_matches_runtime "$engine_path"; then
        echo "Keeping compatible engine: $engine_path"
        return
    fi

    rm -f "$temporary"
    echo "Building $engine_path from $onnx_path"
    "$TRTEXEC" --onnx="$onnx_path" --saveEngine="$temporary" --fp16 --skipInference "$@"
    test -s "$temporary"
    mv -f "$temporary" "$engine_path"
}

if [ "$BUILD_PRIMARY" = 1 ]; then
    # nvstreammux/primary-infer receives the 1, 4, and 8-stream batches.
    build_engine export_yolov8_ds/yolov8s.onnx models/fp16/yolov8s.engine \
        --memPoolSize=workspace:2048M \
        --minShapes=input:1x3x640x640 \
        --optShapes=input:8x3x640x640 \
        --maxShapes=input:8x3x640x640
fi

if [ "$BUILD_SPECIALISTS" = 1 ]; then
    # Python task workers submit one ROI/image per inference request.
    build_engine models/ppe_yolov8n_dynamic.onnx models/fp16/ppe_yolov8n_dynamic_fp16.engine \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:1x3x640x640
    build_engine models/fire_smoke_best.onnx models/fp16/fire_smoke_best_fp16.engine
    build_engine models/v8_n_pose.onnx models/fp16/v8_n_pose_fp16.engine
    build_engine models/plate_detector_fastalpr.onnx models/fp16/plate_detector_fastalpr.engine
    build_engine models/plate_ocr_fastalpr.onnx models/plate_ocr_fastalpr-dynamic.engine \
        --minShapes=input:1x64x128x3 \
        --optShapes=input:1x64x128x3 \
        --maxShapes=input:1x64x128x3
fi

echo "FP16 engine check/build completed for TensorRT ${runtime_series}."
