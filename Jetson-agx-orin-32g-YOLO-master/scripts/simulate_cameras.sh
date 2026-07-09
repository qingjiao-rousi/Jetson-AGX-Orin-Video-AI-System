#!/usr/bin/env bash
set -euo pipefail
# ===========================================================================
# simulate_cameras.sh — 用本地 MP4 文件模拟 8 路 RTSP 监控摄像头
#
# 用法:
#   scripts/simulate_cameras.sh /path/to/videos          # 启动 8 路模拟
#   scripts/simulate_cameras.sh /path/to/videos --limit 4 # 只模拟 4 路
#   scripts/simulate_cameras.sh --stop                    # 停止全部模拟
# ===========================================================================

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT_DIR/.runtime/camera_simulator"
RTSP_PORT="${RTSP_PORT:-8554}"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
MOUNT_PREFIX="${MOUNT_PREFIX:-cam}"
VIDEO_GLOB="${VIDEO_GLOB:-*.mp4}"

usage() {
    cat <<'EOF'
Usage:
  scripts/simulate_cameras.sh VIDEO_DIR [--limit N]    启动模拟（默认 N=8）
  scripts/simulate_cameras.sh --stop                    停止全部模拟

Environment:
  RTSP_PORT=8554        RTSP 服务端口
  RTSP_HOST=127.0.0.1   对外广播的 IP
  MOUNT_PREFIX=cam      挂载点前缀，例如 cam → /cam1 /cam2 ...
  VIDEO_GLOB=*.mp4      视频文件匹配模式

Examples:
  scripts/simulate_cameras.sh /home/nvidia/Desktop/YOLO/video
  scripts/simulate_cameras.sh /home/nvidia/Desktop/YOLO/video --limit 4
  scripts/simulate_cameras.sh --stop
EOF
}

# ---- 停止 ----
stop_all() {
    if [ ! -d "$PID_DIR" ]; then
        echo "No running simulators found (PID dir missing: $PID_DIR)"
        exit 0
    fi

    # 先停 ffmpeg 推流进程
    for pidfile in "$PID_DIR"/ffmpeg_*.pid; do
        [ -f "$pidfile" ] || continue
        pid=$(cat "$pidfile")
        stream=$(basename "$pidfile" .pid | sed 's/ffmpeg_//')
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "Stopped ffmpeg $stream (pid=$pid)"
        fi
        rm -f "$pidfile"
    done
    sleep 0.5

    # 再停 MediaMTX
    if [ -f "$PID_DIR/mediamtx.pid" ]; then
        pid=$(cat "$PID_DIR/mediamtx.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "Stopped MediaMTX (pid=$pid)"
        fi
        rm -f "$PID_DIR/mediamtx.pid"
    fi

    rm -rf "$PID_DIR"
    echo "All camera simulators stopped."
    exit 0
}

# ---- 检查依赖 ----
check_deps() {
    local missing=()
    if ! command -v mediamtx &>/dev/null; then
        missing+=("mediamtx")
    fi
    if ! command -v ffmpeg &>/dev/null; then
        missing+=("ffmpeg")
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        cat <<EOF
Missing dependencies: ${missing[*]}

Installation:
  # MediaMTX (single binary, no dependencies)
  wget https://github.com/bluenviron/mediamtx/releases/download/v1.10.0/mediamtx_v1.10.0_linux_arm64v8.tar.gz
  tar xzf mediamtx_v1.10.0_linux_arm64v8.tar.gz
  sudo cp mediamtx /usr/local/bin/

  # FFmpeg
  sudo apt install -y ffmpeg
EOF
        exit 1
    fi
}

# ---- 主逻辑 ----
main() {
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        usage
        exit 0
    fi

    if [ "${1:-}" = "--stop" ]; then
        stop_all
    fi

    check_deps

    local video_dir="${1:-}"
    if [ -z "$video_dir" ]; then
        echo "ERROR: VIDEO_DIR is required." >&2
        usage
        exit 1
    fi

    if [ ! -d "$video_dir" ]; then
        echo "ERROR: directory not found: $video_dir" >&2
        exit 1
    fi

    local limit=8
    if [ "${2:-}" = "--limit" ] && [ -n "${3:-}" ]; then
        limit="$3"
    fi

    # 收集视频文件
    mapfile -t videos < <(find "$video_dir" -maxdepth 1 -name "$VIDEO_GLOB" -type f \
        | sort | head -n "$limit")
    local count="${#videos[@]}"

    if [ "$count" -eq 0 ]; then
        echo "ERROR: no $VIDEO_GLOB files found in $video_dir" >&2
        exit 1
    fi

    echo "Found $count video(s) in $video_dir"

    # 停止旧的
    if [ -d "$PID_DIR" ] && ls "$PID_DIR"/*.pid &>/dev/null; then
        echo "Stopping previous simulators..."
        for f in "$PID_DIR"/*.pid; do
            [ -f "$f" ] || continue
            kill "$(cat "$f")" 2>/dev/null || true
            rm -f "$f"
        done
    fi
    mkdir -p "$PID_DIR"

    # ---- 启动 MediaMTX ----
    local mtx_config="$PID_DIR/mediamtx.yml"
    cat >"$mtx_config" <<YAML
rtspAddress: :$RTSP_PORT
rtmpAddress: :1935
rtmpAddress: :1935
webrtcAddress: :8889
hlsAddress: :8888
logLevel: info
logDestinations: [stdout]
paths:
YAML
    for i in $(seq 1 "$count"); do
        cat >>"$mtx_config" <<YAML
  ${MOUNT_PREFIX}${i}:
    source: publisher
    sourceOnDemand: no
    disablePublisherOverride: yes
YAML
    done

    echo "Starting MediaMTX on port $RTSP_PORT..."
    mediamtx "$mtx_config" >"$PID_DIR/mediamtx.log" 2>&1 &
    echo $! >"$PID_DIR/mediamtx.pid"
    sleep 2

    if ! kill -0 "$(cat "$PID_DIR/mediamtx.pid")" 2>/dev/null; then
        echo "ERROR: MediaMTX failed to start. Log:"
        cat "$PID_DIR/mediamtx.log"
        exit 1
    fi

    # ---- 为每个视频启动 ffmpeg 推流 ----
    for i in $(seq 0 $((count - 1))); do
        local stream_num=$((i + 1))
        local video="${videos[$i]}"
        local rtsp_url="rtsp://127.0.0.1:$RTSP_PORT/${MOUNT_PREFIX}${stream_num}"
        local logfile="$PID_DIR/ffmpeg_${MOUNT_PREFIX}${stream_num}.log"

        echo "  [$stream_num/$count] $video → $rtsp_url"

        # -stream_loop -1: 无限循环
        # -re:           按原始帧率读取（模拟实时摄像头）
        # -c copy:       不重新编码，零 CPU 开销
        ffmpeg -hide_banner -loglevel error \
            -stream_loop -1 \
            -re \
            -i "$video" \
            -c copy \
            -f rtsp \
            -rtsp_transport tcp \
            "$rtsp_url" \
            >"$logfile" 2>&1 &

        echo $! >"$PID_DIR/ffmpeg_${MOUNT_PREFIX}${stream_num}.pid"
    done

    sleep 1

    # ---- 验证 ----
    local alive=0
    for i in $(seq 1 "$count"); do
        local pidfile="$PID_DIR/ffmpeg_${MOUNT_PREFIX}${i}.pid"
        if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done

    echo ""
    echo "============================================"
    echo "  Simulated Camera Farm Ready"
    echo "============================================"
    echo "  Live streams: $alive/$count"
    echo ""
    for i in $(seq 1 "$count"); do
        echo "  rtsp://$RTSP_HOST:$RTSP_PORT/${MOUNT_PREFIX}$i"
    done
    echo ""
    echo "  MediaMTX WebRTC preview: http://$RTSP_HOST:8889"
    echo "  Stop:  $0 --stop"
    echo "============================================"
    echo ""
    echo "Now run DeepStream against these streams:"
    echo "  RTSP_BASE=rtsp://127.0.0.1:$RTSP_PORT/$MOUNT_PREFIX \\"
    echo "  SOURCE_COUNT=$count \\"
    echo "  OUTPUT_SINK=file \\"
    echo "  scripts/run_rtsp_inproc.sh outputs/rtsp_live"
    echo ""
}

main "$@"
