#!/usr/bin/env python3
"""Calibrate and build an INT8 TensorRT engine for YOLOv8s.

The calibration images must use representative deployment frames. The
default preprocessing matches the project's YOLO input path: RGB, letterbox
to 640x640, CHW layout, and float32 values in [0, 1].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
try:
    from cuda import cudart
except ImportError:
    from cuda.bindings import runtime as cudart


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# 修改 check_cuda 函数（大约在第20行）
def check_cuda(status: object, operation: str) -> None:
    # CUDA 函数通常返回 (err, value) 元组
    if isinstance(status, tuple):
        err = status[0]
    else:
        err = status
    
    if int(err) != 0:
        raise RuntimeError(f"{operation} failed with CUDA status {err}")


def letterbox(image: np.ndarray, target_width: int = 640, target_height: int = 640) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    new_width = max(1, round(source_width * scale))
    new_height = max(1, round(source_height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: list[Path],
        batch_size: int,
        cache_path: Path,
        input_width: int,
        input_height: int,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.input_width = input_width
        self.input_height = input_height
        self.index = 0
        self.host_batch = np.empty((batch_size, 3, input_height, input_width), dtype=np.float32)
        # 修改这里：cudaMalloc 返回 (err, ptr)
        status, self.device_batch = cudart.cudaMalloc(self.host_batch.nbytes)
        check_cuda(status, "cudaMalloc")

    def __del__(self) -> None:
        device_batch = getattr(self, "device_batch", None)
        if device_batch:
            status = cudart.cudaFree(device_batch)
            # 忽略析构时的错误，因为程序可能已经退出
            pass

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        del names
        if self.index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self.index : self.index + self.batch_size]
        if len(batch_paths) < self.batch_size:
            return None

        for slot, path in enumerate(batch_paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read calibration image: {path}")
            image = cv2.cvtColor(
                letterbox(image, self.input_width, self.input_height), cv2.COLOR_BGR2RGB
            )
            self.host_batch[slot] = image.transpose(2, 0, 1).astype(np.float32) / 255.0

        # 修改这里：cudaMemcpy 返回 (err,)
        status = cudart.cudaMemcpy(
            self.device_batch,
            self.host_batch.ctypes.data,
            self.host_batch.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        check_cuda(status, "cudaMemcpy")
        self.index += self.batch_size
        print(f"Calibrating images {self.index}/{len(self.image_paths)}")
        return [int(self.device_batch)]
    
    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.is_file():
            print(f"Using calibration cache: {self.cache_path}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
        print(f"Wrote calibration cache: {self.cache_path}")


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    cache_path: Path,
    image_dir: Path,
    calibration_batch_size: int,
    min_batch_size: int,
    opt_batch_size: int,
    max_batch_size: int,
    workspace_gib: int,
    input_width: int | None,
    input_height: int | None,
) -> None:
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_paths) < calibration_batch_size:
        raise RuntimeError(
            f"need at least {calibration_batch_size} calibration images, found {len(image_paths)}"
        )
    if not min_batch_size <= opt_batch_size <= max_batch_size:
        raise RuntimeError("batch sizes must satisfy min <= opt <= max")
    if calibration_batch_size != opt_batch_size:
        raise RuntimeError(
            "calibration batch size must match the optimization-profile opt batch size"
        )

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    onnx_data = onnx_path.read_bytes()
    if not parser.parse(onnx_data):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse ONNX model:\n{errors}")

    input_tensor = network.get_input(0)
    input_shape = tuple(input_tensor.shape)

    if len(input_shape) != 4:
        raise RuntimeError(f"expected 4D input, got {len(input_shape)}D")

    if input_shape[1] != 3:
        raise RuntimeError(f"expected 3 channels, got {input_shape[1]}")

    model_height, model_width = input_shape[2], input_shape[3]
    h = int(input_height or (model_height if isinstance(model_height, int) and model_height > 0 else 640))
    w = int(input_width or (model_width if isinstance(model_width, int) and model_width > 0 else 640))

    print(f"Input shape: {input_shape}, using spatial size: {h}x{w}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = ImageCalibrator(
        image_paths, calibration_batch_size, cache_path, w, h
    )

    # Keep calibration at the opt batch while allowing DeepStream to submit
    # 1, 4, or 8 frames from the same serialized engine.
    profile = builder.create_optimization_profile()
    min_shape = (min_batch_size, 3, h, w)
    opt_shape = (opt_batch_size, 3, h, w)
    max_shape = (max_batch_size, 3, h, w)
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    if hasattr(config, "set_calibration_profile"):
        config.set_calibration_profile(profile)
    print(
        f"Building INT8 engine from {onnx_path} "
        f"with batch profile min/opt/max={min_batch_size}/{opt_batch_size}/{max_batch_size}"
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the INT8 engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"Wrote INT8 engine: {engine_path}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("export_yolov8_ds/yolov8s.onnx"))
    parser.add_argument("--images", type=Path, default=Path("calibration/yolov8s"))
    parser.add_argument("--cache", type=Path, default=Path("models/yolov8s_calibration.cache"))
    parser.add_argument("--engine", type=Path, default=Path("models/yolov8s_int8.engine"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Calibration batch size; it must match --opt-batch-size.",
    )
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--opt-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--workspace-gib", type=int, default=4)
    parser.add_argument("--input-width", type=int, help="Override the ONNX input width for dynamic models.")
    parser.add_argument("--input-height", type=int, help="Override the ONNX input height for dynamic models.")
    args = parser.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX model not found: {args.onnx}")
    if not args.images.is_dir():
        raise SystemExit(f"Calibration image directory not found: {args.images}")
    build_engine(
        args.onnx,
        args.engine,
        args.cache,
        args.images,
        max(args.batch_size, 1),
        max(args.min_batch_size, 1),
        max(args.opt_batch_size, 1),
        max(args.max_batch_size, 1),
        max(args.workspace_gib, 1),
        args.input_width,
        args.input_height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
