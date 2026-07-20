#!/usr/bin/env python3
"""Export weights from tmp/YOLOv8-pose-master to ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

def prepare_import_paths(project: Path) -> None:
    system = "/usr/lib/python3/dist-packages"
    user = "/home/nvidia/.local/lib/python3.10/site-packages"
    sys.path.insert(0, system)
    sys.path.insert(1, user)
    import numpy  # noqa: F401
    import matplotlib  # noqa: F401
    sys.path.remove(system)
    sys.path.insert(0, user)
    sys.path.append(system)
    sys.path.insert(0, str(project))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/v8_n_pose.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    project = Path("tmp/YOLOv8-pose-master").resolve()
    prepare_import_paths(project)
    import torch
    from nets import nn

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"checkpoint does not contain a torch model: {type(model)!r}")
    model = model.float().eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported custom pose ONNX: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
