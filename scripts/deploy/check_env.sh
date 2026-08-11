#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [ -f scripts/deploy/env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/deploy/env.sh >/dev/null
fi

check() {
    local label="$1"
    shift
    echo ""
    echo "== $label =="
    "$@"
    local code=$?
    if [ "$code" -eq 0 ]; then
        echo "[OK] $label"
    else
        echo "[FAIL] $label"
    fi
    return "$code"
}

failures=0

check "Python" python3 --version || failures=$((failures + 1))
check "DeepStream" deepstream-app --version-all || failures=$((failures + 1))
check "GStreamer" gst-inspect-1.0 --version || failures=$((failures + 1))
check "nvstreammux" gst-inspect-1.0 nvstreammux || failures=$((failures + 1))
check "nvinfer" gst-inspect-1.0 nvinfer || failures=$((failures + 1))
check "nvtracker" gst-inspect-1.0 nvtracker || failures=$((failures + 1))
check "nvdsosd" gst-inspect-1.0 nvdsosd || failures=$((failures + 1))
check "nvvideoconvert" gst-inspect-1.0 nvvideoconvert || failures=$((failures + 1))
check "nvv4l2decoder" gst-inspect-1.0 nvv4l2decoder || failures=$((failures + 1))
check "PyYAML" python3 -c "import yaml; print('PyYAML OK')" || failures=$((failures + 1))
check "pyds" python3 -c "import pyds; print('pyds OK')" || failures=$((failures + 1))
check "Project import" python3 -c "from app.bootstrap import create_application; print('project import OK')" || failures=$((failures + 1))

echo ""
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "[WARN] nvcc not found. Install cuda-nvcc-12-6 or cuda-toolkit-12-6 if you need model/parser builds."
fi

if command -v trtexec >/dev/null 2>&1; then
    trtexec --version
else
    echo "[WARN] trtexec not found. Install libnvinfer-bin to build TensorRT engines."
fi

echo ""
echo "Failures: $failures"
exit "$failures"
