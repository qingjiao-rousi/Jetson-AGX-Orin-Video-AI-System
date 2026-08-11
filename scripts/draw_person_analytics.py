#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw ROI and line analytics overlays on an MP4.")
    parser.add_argument("input_video", type=Path, help="Input MP4 path.")
    parser.add_argument("config_yaml", type=Path, help="Analytics config YAML path.")
    parser.add_argument("output_video", type=Path, help="Output MP4 path with overlays.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config_yaml)
    draw_overlays(args.input_video, config, args.output_video)
    print(f"Wrote analytics overlay video: {args.output_video}")
    return 0


def draw_overlays(input_video: Path, config: dict[str, Any], output_video: Path) -> None:
    if not input_video.exists():
        raise FileNotFoundError(input_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output video: {output_video}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _draw_rois(frame, config.get("rois", []))
            _draw_lines(frame, config.get("lines", []))
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def _draw_rois(frame, rois: list[dict[str, Any]]) -> None:
    for roi in rois:
        x, y, width, height = (int(value) for value in roi["rect"])
        color = tuple(int(value) for value in roi.get("color", [0, 255, 255]))
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            str(roi["id"]),
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_lines(frame, lines: list[dict[str, Any]]) -> None:
    for line in lines:
        x1, y1, x2, y2 = (int(value) for value in line["points"])
        color = tuple(int(value) for value in line.get("color", [0, 0, 255]))
        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            str(line["id"]),
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Analytics config must be a YAML mapping")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
