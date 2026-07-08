from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.adapters.config_loader import load_settings
from app.adapters.runtime_overrides import apply_runtime_overrides
from app.bootstrap import create_application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-stream DeepStream application")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/app/app.yaml"),
        help="Path to the application config file.",
    )
    parser.add_argument("--input-video", type=Path, help="Local MP4 input path.")
    parser.add_argument("--output-video", type=Path, help="Output MP4 path.")
    parser.add_argument("--output-json", type=Path, help="Output JSONL path.")
    parser.add_argument("--output-width", type=int, help="Encoded output width.")
    parser.add_argument("--output-height", type=int, help="Encoded output height.")
    parser.add_argument("--confidence-threshold", type=float, help="YOLO pre-cluster threshold.")
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Keep all classes instead of filtering to person only.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the dashboard server for batch processing.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="Directory for per-run generated DeepStream config files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    settings = apply_runtime_overrides(
        settings,
        input_video=args.input_video,
        output_video=args.output_video,
        output_json=args.output_json,
        output_width=args.output_width,
        output_height=args.output_height,
        confidence_threshold=args.confidence_threshold,
        person_only=not args.all_classes,
        enable_web=False if args.no_web else None,
        runtime_dir=args.runtime_dir,
    )
    app = create_application(args.config, settings=settings)

    logging.info("application starting")
    try:
        if app.dashboard_server is not None:
            app.dashboard_server.start()
        app.orchestrator.start()
        app.orchestrator.run_forever()
        return 0
    except KeyboardInterrupt:
        logging.info("shutdown requested by user")
        return 0
    except Exception:
        logging.exception("application crashed")
        return 1
    finally:
        app.orchestrator.stop()
        if app.dashboard_server is not None:
            app.dashboard_server.stop()
        logging.info("application stopped")


if __name__ == "__main__":
    raise SystemExit(main())
