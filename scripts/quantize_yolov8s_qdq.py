#!/usr/bin/env python3
"""Create an explicitly quantized Q/DQ ONNX model for YOLOv8s."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import shape_inference
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


def letterbox(image: np.ndarray, size: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


class ImageDataReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str) -> None:
        self.image_paths = image_paths
        self.input_name = input_name
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.image_paths):
            return None
        path = self.image_paths[self.index]
        self.index += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read calibration image: {path}")
        image = cv2.cvtColor(letterbox(image), cv2.COLOR_BGR2RGB)
        tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        if self.index % 50 == 0 or self.index == len(self.image_paths):
            print(f"Calibrating images {self.index}/{len(self.image_paths)}")
        return {self.input_name: tensor[None, ...]}

    def rewind(self) -> None:
        self.index = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("models/yolov8s.onnx"))
    parser.add_argument("--images", type=Path, default=Path("calibration/yolov8s"))
    parser.add_argument("--output", type=Path, default=Path("models/yolov8s_int8_qdq.onnx"))
    parser.add_argument(
        "--preprocess-output",
        type=Path,
        help="Write an ONNX Runtime preprocessed model here before quantization.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"ONNX model not found: {args.input}")
    image_paths = sorted(
        path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise SystemExit(f"no calibration images found in {args.images}")

    model_path = args.input
    if args.preprocess_output:
        if args.preprocess_output == args.input:
            raise SystemExit("--preprocess-output must not overwrite --input")
        args.preprocess_output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Preprocessing ONNX graph: {args.input} -> {args.preprocess_output}")
        # YOLOv8's dynamic detection head cannot complete ORT symbolic shape inference.
        quant_pre_process(
            input_model=str(args.input),
            output_model_path=str(args.preprocess_output),
            skip_symbolic_shape=True,
        )
        model_path = args.preprocess_output

    model = onnx.load(str(model_path))
    input_name = model.graph.input[0].name
    print(f"Input: {input_name}")
    print(f"Calibration images: {len(image_paths)}")
    print(f"Output: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageDataReader(image_paths, input_name)
    quantize_static(
        model_input=str(model_path),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "QuantizeBias": False,
        },
    )
    quantized_model = onnx.load(str(args.output))
    quantized_model = shape_inference.infer_shapes(quantized_model)
    onnx.save(quantized_model, str(args.output))
    print("Added ONNX shape metadata")
    print(f"Wrote Q/DQ model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
