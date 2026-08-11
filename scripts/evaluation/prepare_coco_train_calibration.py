#!/usr/bin/env python3
"""Write a deterministic COCO train2017 image manifest and download URL list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.application.coco_calibration import select_calibration_images


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COCO_TRAIN_URL = "https://images.cocodataset.org/train2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="COCO instances_train2017.json path.")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration/coco_train504"))
    parser.add_argument("--count", type=int, default=504, help="Selected images; default is divisible by batch 8.")
    parser.add_argument("--batch-size", type=int, default=8, help="Calibration batch size used to validate --count.")
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    annotation_path, output_dir = resolve(args.annotations), resolve(args.output_dir)
    if not annotation_path.is_file():
        raise SystemExit(f"missing COCO train annotations: {annotation_path}")
    if args.count <= 0 or args.batch_size <= 0:
        raise SystemExit("--count and --batch-size must be greater than zero")
    if args.count % args.batch_size:
        raise SystemExit(
            f"--count ({args.count}) must be divisible by --batch-size ({args.batch_size}); "
            "otherwise the current calibrator ignores the last incomplete batch"
        )
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list):
        raise SystemExit("COCO annotation file has no images array")
    selected = select_calibration_images(images, count=args.count, seed=args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "purpose": "TensorRT INT8 calibration only; not training and not COCO val2017 evaluation",
        "annotations": str(annotation_path),
        "count": args.count,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "images": [
            {"id": int(item["id"]), "file_name": str(item["file_name"]), "url": f"{COCO_TRAIN_URL}/{item['file_name']}"}
            for item in selected
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "download_urls.txt").write_text(
        "".join(f"{item['url']}\n" for item in manifest["images"]), encoding="utf-8"
    )
    print(f"Wrote {args.count} COCO train2017 calibration URLs: {output_dir / 'download_urls.txt'}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
