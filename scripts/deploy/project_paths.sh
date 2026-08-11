#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_PARENT="$(cd "$PROJECT_ROOT/.." && pwd)"

DEPLOY_ENV="${DEPLOY_ENV:-/etc/campus-surveillance/campus-surveillance.env}"
if [ -f "$DEPLOY_ENV" ]; then
    # shellcheck disable=SC1090
    source "$DEPLOY_ENV"
fi

PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-$PROJECT_ROOT}"
VIDEO_DIR="${VIDEO_DIR:-$PROJECT_PARENT/video}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_ROOT/.runtime}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

export PROJECT_ROOT VIDEO_DIR OUTPUT_ROOT RUNTIME_ROOT LOG_ROOT DEPLOY_ENV
