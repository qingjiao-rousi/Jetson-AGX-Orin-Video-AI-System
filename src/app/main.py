from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.bootstrap import create_application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-stream DeepStream application")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/app/app.yaml"),
        help="Path to the application config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_application(args.config)

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
