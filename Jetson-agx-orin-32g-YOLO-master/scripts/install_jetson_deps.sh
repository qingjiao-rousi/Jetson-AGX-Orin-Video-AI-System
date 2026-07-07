#!/usr/bin/env bash
set -euo pipefail

DEEPSTREAM_DIR="${DEEPSTREAM_DIR:-/opt/nvidia/deepstream/deepstream-7.1}"
WORK_DIR="${WORK_DIR:-$HOME/src}"
PYDS_TAG="${PYDS_TAG:-v1.2.0}"

echo "This script installs Jetson system dependencies for this project."
echo "It requires sudo privileges and internet access."
echo ""

sudo apt-get update
sudo apt-get install -y \
    cuda-nvcc-12-6 \
    libnvinfer-bin \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gst-1.0 \
    python-gi-dev \
    git \
    cmake \
    g++ \
    build-essential \
    libglib2.0-dev \
    libglib2.0-dev-bin \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r requirements.txt

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d deepstream_python_apps ]; then
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
fi

cd deepstream_python_apps
git fetch --tags
git checkout "$PYDS_TAG"
git submodule update --init --recursive
cd bindings
rm -rf build
mkdir build
cd build

cmake .. \
    -DPYTHON_MAJOR_VERSION=3 \
    -DPYTHON_MINOR_VERSION=10 \
    -DPIP_PLATFORM=linux_aarch64 \
    -DDS_PATH="$DEEPSTREAM_DIR"

make -j"$(nproc)"
python3 -m pip install --user ./pyds-*.whl

echo ""
echo "Installed Jetson dependencies. Re-open the shell or run:"
echo "  source scripts/env.sh"
echo "  bash scripts/check_env.sh"
