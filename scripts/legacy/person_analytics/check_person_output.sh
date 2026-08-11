#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR" || exit 1

VIDEO_PATH="${1:-outputs/person_detect.mp4}"
JSON_PATH="${2:-outputs/results.jsonl}"
EXPECTED_WIDTH="${EXPECTED_WIDTH:-1280}"
EXPECTED_HEIGHT="${EXPECTED_HEIGHT:-720}"
REQUIRE_TRACKS="${REQUIRE_TRACKS:-0}"

failures=0

check_file() {
    local label="$1"
    local path="$2"
    if [ -s "$path" ]; then
        echo "[OK] $label exists: $path"
    else
        echo "[FAIL] $label missing or empty: $path"
        failures=$((failures + 1))
    fi
}

check_file "Video" "$VIDEO_PATH"
check_file "JSONL" "$JSON_PATH"

if command -v gst-discoverer-1.0 >/dev/null 2>&1 && [ -s "$VIDEO_PATH" ]; then
    discover="$(gst-discoverer-1.0 "$VIDEO_PATH" 2>&1)"
    echo "$discover"
    echo "$discover" | grep -q "Width: $EXPECTED_WIDTH" || failures=$((failures + 1))
    echo "$discover" | grep -q "Height: $EXPECTED_HEIGHT" || failures=$((failures + 1))
else
    echo "[WARN] gst-discoverer-1.0 unavailable or video missing; skipping video inspection"
fi

if [ -s "$JSON_PATH" ]; then
    python3 - "$JSON_PATH" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = 0
detection_lines = 0
bad_classes = set()
bad_confidence = 0
tracker_only_confidence = 0
bad_boxes = 0
track_counts = {}

with path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        lines += 1
        payload = json.loads(line)
        detections = payload.get("detections", [])
        tracks = payload.get("tracks", [])
        if detections:
            detection_lines += 1
        for track in tracks:
            track_id = int(track.get("track_id", -1))
            if track_id >= 0:
                track_counts[track_id] = track_counts.get(track_id, 0) + 1
        for det in detections:
            class_id = det.get("class_id")
            class_name = det.get("class_name")
            if class_id != 0 or class_name != "person":
                bad_classes.add((class_id, class_name))
            confidence = float(det.get("confidence", 0.0))
            # DeepStream tracker marks propagated, non-detector objects as -0.1.
            if confidence == -0.1:
                tracker_only_confidence += 1
                continue
            if confidence < 0.0 or confidence > 1.0:
                bad_confidence += 1
            bbox = det.get("bbox", {})
            if float(bbox.get("width", 0.0)) <= 0.0 or float(bbox.get("height", 0.0)) <= 0.0:
                bad_boxes += 1

print(f"JSONL lines: {lines}")
print(f"Lines with detections: {detection_lines}")
print(f"Unique track IDs: {len(track_counts)}")
print(f"Tracker-propagated detections (confidence=-0.1): {tracker_only_confidence}")
if bad_classes:
    print(f"[FAIL] Non-person classes found: {sorted(bad_classes)}")
    raise SystemExit(1)
if bad_confidence:
    print(f"[FAIL] Detections with confidence outside 0..1: {bad_confidence}")
    raise SystemExit(1)
if bad_boxes:
    print(f"[FAIL] Detections with empty boxes: {bad_boxes}")
    raise SystemExit(1)
if bool(int(os.environ.get("REQUIRE_TRACKS", "0"))):
    stable_tracks = {track_id: count for track_id, count in track_counts.items() if count >= 2}
    if not track_counts:
        print("[FAIL] No tracks found")
        raise SystemExit(1)
    if not stable_tracks:
        print("[FAIL] No track_id appears across multiple frames")
        raise SystemExit(1)
    print(f"[OK] Stable tracks found: {len(stable_tracks)}")
print("[OK] JSONL detections are person-only with valid boxes/confidence")
PY
    code=$?
    if [ "$code" -ne 0 ]; then
        failures=$((failures + 1))
    fi
fi

echo ""
echo "Failures: $failures"
exit "$failures"
