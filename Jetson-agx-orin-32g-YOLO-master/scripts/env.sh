#!/usr/bin/env bash
# Source this file before running the project on Jetson:
#   source scripts/env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEEPSTREAM_DIR="${DEEPSTREAM_DIR:-/opt/nvidia/deepstream/deepstream-7.1}"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/usr/local/cuda/bin:/usr/src/tensorrt/bin:$DEEPSTREAM_DIR/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib64:$DEEPSTREAM_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GST_PLUGIN_PATH="$DEEPSTREAM_DIR/lib/gst-plugins:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/deepstream${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"

echo "Project env loaded:"
echo "  ROOT_DIR=$ROOT_DIR"
echo "  DEEPSTREAM_DIR=$DEEPSTREAM_DIR"
