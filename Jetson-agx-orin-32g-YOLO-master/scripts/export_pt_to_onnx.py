#!/usr/bin/env python3
"""Export a YOLO .pt artifact to ONNX on the Jetson host.

The host currently has system NumPy/Matplotlib and user-site ONNX/protobuf.
Import order is controlled here so those compatible packages can coexist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def prepare_import_paths() -> None:
    system = "/usr/lib/python3/dist-packages"
    user = "/home/nvidia/.local/lib/python3.10/site-packages"
    sys.path.insert(0, system)
    sys.path.insert(1, user)
    import numpy  # noqa: F401
    import matplotlib  # noqa: F401

    if system in sys.path:
        sys.path.remove(system)
    sys.path.insert(0, user)
    sys.path.append(system)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    prepare_import_paths()
    from ultralytics import YOLO

    YOLO(str(args.model)).export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        dynamic=False,
        simplify=True,
        opset=17,
        device="cpu",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
