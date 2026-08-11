#!/usr/bin/env python3
"""Run the PPE batch-1/4/8 micro-batch experiment on one fixed system config."""

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
    parser.add_argument("--base-config", type=Path, default=Path("configs/app/app_multifile_8_primary_int8.yaml"))
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--wait-ms", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--run-seconds", type=float, default=180.0)
    parser.add_argument("--sink", choices=("fake", "file"), default="fake")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ppe_microbatch"))
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


def configure_ppe(base: dict[str, Any], batch_size: int, wait_ms: int) -> dict[str, Any]:
    config = clone_config(base)
    models = config.setdefault("models", {})
    helmet = models.get("helmet")
    if not isinstance(helmet, dict):
        raise ValueError("base configuration must define models.helmet")
    if batch_size > 1:
        helmet["engine"] = f"models/fp16/ppe_yolov8n_dynamic_fp16_b{batch_size}.engine"
    tasks = config.setdefault("model_tasks", {})
    task = tasks.get("helmet")
    if not isinstance(task, dict):
        raise ValueError("base configuration must define model_tasks.helmet")
    task["micro_batch_size"] = batch_size
    task["micro_batch_wait_ms"] = 0 if batch_size == 1 else wait_ms
    config.setdefault("app", {})["app_name"] = f"ppe-microbatch-{batch_size}"
    return config


def numeric(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in sorted({int(run["batch_size"]) for run in runs}):
        group = [run for run in runs if run["batch_size"] == batch_size and run.get("summary")]

        def avg(*keys: str) -> float | None:
            values = [numeric(run["summary"], *keys) for run in group]
            values = [value for value in values if value is not None]
            return round(mean(values), 3) if values else None

        rows.append(
            {
                "ppe_micro_batch_size": batch_size,
                "successful_repetitions": len(group),
                "average_processing_fps": avg("runtime", "average_processing_fps"),
                "average_ppe_batch_size": avg("drop_and_queue_stats", "helmet_worker", "batch_size", "average"),
                "ppe_queue_wait_p50_ms": avg("drop_and_queue_stats", "helmet_worker", "queue_wait_ms", "p50"),
                "ppe_queue_wait_p95_ms": avg("drop_and_queue_stats", "helmet_worker", "queue_wait_ms", "p95"),
                "ppe_task_latency_p50_ms": avg("drop_and_queue_stats", "helmet_worker", "task_latency_ms", "p50"),
                "ppe_task_latency_p95_ms": avg("drop_and_queue_stats", "helmet_worker", "task_latency_ms", "p95"),
                "ppe_batch_inference_p50_ms": avg("drop_and_queue_stats", "helmet_worker", "inference_ms", "p50"),
                "ppe_batch_inference_p95_ms": avg("drop_and_queue_stats", "helmet_worker", "inference_ms", "p95"),
                "ppe_processed": avg("drop_and_queue_stats", "helmet_worker", "processed"),
                "ppe_missing_frames": avg("drop_and_queue_stats", "helmet_worker", "missing_frames"),
                "ppe_task_dropped": avg("drop_and_queue_stats", "task_buffer_by_task", "helmet", "dropped"),
                "frame_store_dropped": avg("drop_and_queue_stats", "frame_store_dropped"),
                "helmet_violation_events": avg("predictions", "event_counts", "helmet_violation"),
            }
        )
    return rows


def event_consistency(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(int(run["batch_size"]), int(run["repetition"])): run for run in runs if run.get("summary")}
    comparisons: list[dict[str, Any]] = []
    for (batch_size, repetition), candidate in sorted(by_key.items()):
        if batch_size == 1:
            continue
        baseline = by_key.get((1, repetition))
        if baseline is None:
            continue
        left = {tuple(item) for item in baseline["summary"]["predictions"]["event_signatures"]}
        right = {tuple(item) for item in candidate["summary"]["predictions"]["event_signatures"]}
        comparisons.append(
            {
                "repetition": repetition,
                "baseline_batch_size": 1,
                "candidate_batch_size": batch_size,
                "matched_events": len(left & right),
                "batch1_only_events": len(left - right),
                "candidate_only_events": len(right - left),
            }
        )
    return comparisons


def main() -> int:
    args = parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or any(value not in {1, 4, 8} for value in batch_sizes):
        raise SystemExit("--batch-sizes must be a non-empty subset of 1,4,8")
    if args.repetitions <= 0 or args.wait_ms < 0:
        raise SystemExit("--repetitions must be positive and --wait-ms must not be negative")

    base = load_config(args.base_config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = resolve(args.output_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        config = configure_ppe(base, batch_size, args.wait_ms)
        for repetition in range(1, args.repetitions + 1):
            output_dir = root / f"ppe_b{batch_size}_run{repetition}"
            output_dir.mkdir(parents=True, exist_ok=True)
            config_path = output_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            record: dict[str, Any] = {
                "batch_size": batch_size,
                "wait_ms": 0 if batch_size == 1 else args.wait_ms,
                "repetition": repetition,
                "config": str(config_path),
                "output_dir": str(output_dir),
                "execute": args.execute,
            }
            if args.execute:
                engine = resolve(Path(config["models"]["helmet"]["engine"]))
                if not engine.is_file():
                    raise SystemExit(f"missing PPE engine for batch {batch_size}: {engine}")
                environment = os.environ | {
                    "OUTPUT_SINK": args.sink,
                    "ENABLE_TEGRASTATS": "1",
                    "RUN_SECONDS": str(args.run_seconds),
                }
                completed = subprocess.run(
                    ["scripts/deploy/run_multistream.sh", str(config_path), str(output_dir)],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                )
                record["exit_code"] = completed.returncode
                if completed.returncode == 0:
                    record["summary"] = summarize(f"ppe_b{batch_size}_run{repetition}", output_dir, 5)
                    (output_dir / "summary.json").write_text(
                        json.dumps(record["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
            runs.append(record)

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "executed": args.execute,
        "comparison": "PPE micro-batch 1 vs 4 vs 8; primary INT8 and other specialist engines unchanged",
        "runs": runs,
        "aggregates": aggregate(runs) if args.execute else [],
        "event_consistency": event_consistency(runs) if args.execute else [],
    }
    path = root / "matrix_summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote PPE micro-batch {'results' if args.execute else 'plan'}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
