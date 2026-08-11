#!/usr/bin/env bash
set -euo pipefail

# Build the third-party DeepStream YOLO parser locally. The generated .so is
# intentionally ignored because it is coupled to the target Jetson software.
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_DIR="${PARSER_SOURCE_DIR:-$ROOT_DIR/.runtime/DeepStream-Yolo}"
REPOSITORY="${DEEPSTREAM_YOLO_REPOSITORY:-https://github.com/marcoslucianops/DeepStream-Yolo.git}"
REVISION="${DEEPSTREAM_YOLO_REVISION:-master}"
OUTPUT_DIR="$ROOT_DIR/custom_libs/nvdsinfer_custom_impl_Yolo"
CUDA_VER="${CUDA_VER:-}"

if [ -z "$CUDA_VER" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p')"
fi
CUDA_VER="${CUDA_VER:-12.6}"

if [ ! -d "$SOURCE_DIR/.git" ]; then
    mkdir -p "$(dirname "$SOURCE_DIR")"
    git clone --depth 1 "$REPOSITORY" "$SOURCE_DIR"
fi

git -C "$SOURCE_DIR" fetch --depth 1 origin "$REVISION"
git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD

make -C "$SOURCE_DIR/nvdsinfer_custom_impl_Yolo" CUDA_VER="$CUDA_VER"

mkdir -p "$OUTPUT_DIR"
install -m 0755 \
    "$SOURCE_DIR/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so" \
    "$OUTPUT_DIR/libnvdsinfer_custom_impl_Yolo.so"

echo "Built parser: $OUTPUT_DIR/libnvdsinfer_custom_impl_Yolo.so"
echo "Source revision: $(git -C "$SOURCE_DIR" rev-parse HEAD)"
sha256sum "$OUTPUT_DIR/libnvdsinfer_custom_impl_Yolo.so"
