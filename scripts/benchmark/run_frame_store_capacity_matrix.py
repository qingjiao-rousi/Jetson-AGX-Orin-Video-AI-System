#!/usr/bin/env python3
"""Compare shared and per-stream FrameStore capacities on one fixed workload."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from summarize_precision_run import summarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/app/app_multifile_8_primary_int8_isolated_tasks.yaml"),
    )
    parser.add_argument("--per-stream-capacities", default="16,32,64")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--run-seconds", type=float, default=0.0)
    parser.add_argument("--sink", choices=("fake", "file"), default="fake")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/frame_store_capacity"))
    parser.add_argument("--execute", action="store_true", help="Run the matrix; default writes only the plan.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with resolve(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("base configuration must be a YAML mapping")
    return payload


def clone_config(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def enabled_source_count(config: dict[str, Any]) -> int:
    sources = config.get("sources", ())
    return sum(1 for source in sources if not isinstance(source, dict) or source.get("enabled", True))


def configure_frame_store(
    base: dict[str, Any], *, mode: str, per_stream_capacity: int, source_count: int
) -> dict[str, Any]:
    config = clone_config(base)
    optimization = config.setdefault("optimization", {})
    total_capacity = per_stream_capacity * source_count
    optimization["frame_store_max_size"] = total_capacity
    if mode == "per_stream":
        optimization["frame_store_per_stream_capacity"] = per_stream_capacity
    elif mode == "shared":
        optimization.pop("frame_store_per_stream_capacity", None)
    else:
        raise ValueError(f"unsupported FrameStore mode: {mode}")
    config.setdefault("app", {})["app_name"] = (
        f"frame-store-{mode}-{per_stream_capacity}x{source_count}"
    )
    return config


def numeric(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("summary"):
            groups.setdefault((str(run["mode"]), int(run["per_stream_capacity"])), []).append(run)
    rows: list[dict[str, Any]] = []
    for (mode, capacity), group in sorted(groups.items()):
        def avg(*keys: str) -> float | None:
            values = [numeric(run["summary"], *keys) for run in group]
            usable = [value for value in values if value is not None]
            return round(mean(usable), 3) if usable else None

        row: dict[str, Any] = {
            "mode": mode,
            "per_stream_capacity": capacity if mode == "per_stream" else None,
            "total_capacity": group[0]["total_capacity"],
            "successful_repetitions": len(group),
            "average_processing_fps": avg("runtime", "average_processing_fps"),
            "end_to_end_p95_ms": avg("latency", "end_to_end_p95_ms"),
            "frame_store_evicted": avg("drop_and_queue_stats", "frame_store", "evicted"),
            "frame_store_evicted_global": avg("drop_and_queue_stats", "frame_store", "evicted_global"),
            "frame_store_evicted_per_stream": avg("drop_and_queue_stats", "frame_store", "evicted_per_stream"),
        }
        for task in ("helmet", "pose", "fire_smoke"):
            row[f"{task}_missing_frames"] = avg(
                "drop_and_queue_stats", "workers", task, "missing_frames"
            )
            row[f"{task}_frame_age_p95_ms"] = avg(
                "drop_and_queue_stats", "frame_store", "by_consumer", task, "frame_age_ms", "p95"
            )
            row[f"{task}_task_latency_p95_ms"] = avg(
                "drop_and_queue_stats", "workers", task, "task_latency_ms", "p95"
            )
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    capacities = [int(value) for value in args.per_stream_capacities.split(",") if value.strip()]
    if not capacities or any(value <= 0 for value in capacities):
        raise SystemExit("--per-stream-capacities must contain positive integers")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    base = load_config(args.base_config)
    source_count = enabled_source_count(base)
    if source_count <= 0:
        raise SystemExit("base configuration must have at least one enabled source")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = resolve(args.output_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    variants = [("shared", capacities[0])] + [("per_stream", value) for value in capacities]
    runs: list[dict[str, Any]] = []
    for mode, capacity in variants:
        config = configure_frame_store(
            base, mode=mode, per_stream_capacity=capacity, source_count=source_count
        )
        for repetition in range(1, args.repetitions + 1):
            output_dir = root / f"{mode}_{capacity}_run{repetition}"
            output_dir.mkdir(parents=True, exist_ok=True)
            config_path = output_dir / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            record: dict[str, Any] = {
                "mode": mode,
                "per_stream_capacity": capacity,
                "total_capacity": capacity * source_count,
                "repetition": repetition,
                "config": str(config_path),
                "output_dir": str(output_dir),
                "execute": args.execute,
            }
            if args.execute:
                completed = subprocess.run(
                    ["scripts/deploy/run_multistream.sh", str(config_path), str(output_dir)],
                    cwd=PROJECT_ROOT,
                    env=os.environ | {
                        "OUTPUT_SINK": args.sink,
                        "ENABLE_TEGRASTATS": "1",
                        "RUN_SECONDS": str(args.run_seconds),
                    },
                    check=False,
                )
                record["exit_code"] = completed.returncode
                if completed.returncode == 0:
                    record["summary"] = summarize(f"{mode}_{capacity}_run{repetition}", output_dir, 5)
                    (output_dir / "summary.json").write_text(
                        json.dumps(record["summary"], ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            runs.append(record)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "executed": args.execute,
        "comparison": "FrameStore shared baseline versus per-stream capacities; scheduling and model settings unchanged",
        "source_count": source_count,
        "runs": runs,
        "aggregates": aggregate(runs) if args.execute else [],
    }
    path = root / "matrix_summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote FrameStore capacity {'results' if args.execute else 'plan'}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
