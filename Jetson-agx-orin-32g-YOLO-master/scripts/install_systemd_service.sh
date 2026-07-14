#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/project_paths.sh"

SERVICE_NAME="${SERVICE_NAME:-campus-surveillance}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-$(id -un)}}"
SERVICE_GROUP="${SERVICE_GROUP:-$(id -gn "$SERVICE_USER" 2>/dev/null || echo "$SERVICE_USER")}"
ENV_DIR="${ENV_DIR:-/etc/campus-surveillance}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
LOGROTATE_DIR="${LOGROTATE_DIR:-/etc/logrotate.d}"

require_sudo() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "This installer needs sudo for /etc writes."
        echo "Run: sudo -E $0"
        exit 1
    fi
}

render_file() {
    local src="$1"
    local dst="$2"
    sed \
        -e "s|@PROJECT_ROOT@|$PROJECT_ROOT|g" \
        -e "s|@PROJECT_PARENT@|$PROJECT_PARENT|g" \
        -e "s|@OUTPUT_ROOT@|$OUTPUT_ROOT|g" \
        -e "s|@RUNTIME_ROOT@|$RUNTIME_ROOT|g" \
        -e "s|@LOG_ROOT@|$LOG_ROOT|g" \
        -e "s|@SERVICE_USER@|$SERVICE_USER|g" \
        -e "s|@SERVICE_GROUP@|$SERVICE_GROUP|g" \
        "$src" > "$dst"
}

require_sudo

mkdir -p "$ENV_DIR" "$SYSTEMD_DIR" "$LOGROTATE_DIR" "$LOG_ROOT" "$OUTPUT_ROOT" "$RUNTIME_ROOT"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$LOG_ROOT" "$OUTPUT_ROOT" "$RUNTIME_ROOT"

env_target="$ENV_DIR/campus-surveillance.env"
if [ ! -f "$env_target" ]; then
    render_file "$ROOT_DIR/deploy/campus-surveillance.env" "$env_target"
elif grep -q '@PROJECT_ROOT@\|@OUTPUT_ROOT@\|@RUNTIME_ROOT@\|@LOG_ROOT@' "$env_target"; then
    render_file "$ROOT_DIR/deploy/campus-surveillance.env" "$env_target"
fi

# Keep the installed env tied to the current checkout unless the user edits it later.
sed -i \
    -e "s|^PROJECT_ROOT_OVERRIDE=.*|PROJECT_ROOT_OVERRIDE=$PROJECT_ROOT|g" \
    -e "s|^VIDEO_DIR=.*|VIDEO_DIR=$PROJECT_PARENT/video|g" \
    -e "s|^OUTPUT_ROOT=.*|OUTPUT_ROOT=$OUTPUT_ROOT|g" \
    -e "s|^RUNTIME_ROOT=.*|RUNTIME_ROOT=$RUNTIME_ROOT|g" \
    -e "s|^LOG_ROOT=.*|LOG_ROOT=$LOG_ROOT|g" \
    -e "s|^SERVICE_OUTPUT_DIR=.*|SERVICE_OUTPUT_DIR=$OUTPUT_ROOT/campus_surveillance|g" \
    "$env_target"

render_file "$ROOT_DIR/deploy/campus-surveillance.service" "$SYSTEMD_DIR/$SERVICE_NAME.service"
render_file "$ROOT_DIR/deploy/campus-surveillance-cleanup.service" "$SYSTEMD_DIR/$SERVICE_NAME-cleanup.service"
cp "$ROOT_DIR/deploy/campus-surveillance-cleanup.timer" "$SYSTEMD_DIR/$SERVICE_NAME-cleanup.timer"
render_file "$ROOT_DIR/deploy/logrotate/campus-surveillance" "$LOGROTATE_DIR/$SERVICE_NAME"

chmod 0644 \
    "$SYSTEMD_DIR/$SERVICE_NAME.service" \
    "$SYSTEMD_DIR/$SERVICE_NAME-cleanup.service" \
    "$SYSTEMD_DIR/$SERVICE_NAME-cleanup.timer" \
    "$LOGROTATE_DIR/$SERVICE_NAME" \
    "$env_target"
systemctl daemon-reload

echo "Installed systemd service: $SYSTEMD_DIR/$SERVICE_NAME.service"
echo "Installed cleanup service: $SYSTEMD_DIR/$SERVICE_NAME-cleanup.service"
echo "Installed cleanup timer: $SYSTEMD_DIR/$SERVICE_NAME-cleanup.timer"
echo "Installed env file: $env_target"
echo "Installed logrotate config: $LOGROTATE_DIR/$SERVICE_NAME"
echo "Service user/group: $SERVICE_USER:$SERVICE_GROUP"
echo ""
echo "Next commands:"
echo "  sudo systemctl start $SERVICE_NAME"
echo "  sudo systemctl status $SERVICE_NAME"
echo "  sudo systemctl enable $SERVICE_NAME"
echo "  sudo systemctl enable --now $SERVICE_NAME-cleanup.timer"
