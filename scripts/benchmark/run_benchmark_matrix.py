#!/usr/bin/env python3
"""Create or run the reproducible primary-YOLO FP16/INT8 benchmark matrix.

The default is a dry run. Pass ``--execute`` only after checking the generated
plan and ensuring that both model configurations and all local videos exist.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from summarize_precision_run import summarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16-config", type=Path, default=Path("configs/app/app_multifile_8.yaml"))
    parser.add_argument(
        "--primary-int8-config",
        type=Path,
        default=Path("configs/app/app_multifile_8_primary_int8.yaml"),
        help="Configuration where only the primary YOLO engine is INT8.",
    )
    parser.add_argument(
        "--primary-int8-engine",
        type=Path,
        help="Optional candidate primary INT8 engine path; does not modify the base YAML.",
    )
    parser.add_argument("--fp16-confidence-threshold", type=float, help="Primary FP16 pre-cluster threshold for every run.")
    parser.add_argument("--int8-confidence-threshold", type=float, help="Primary INT8 pre-cluster threshold for every run.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--stream-counts", default="1,4,8")
    parser.add_argument("--sinks", default="fake,file")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--run-seconds", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true", help="Run the matrix; default only writes a plan.")
    return parser.parse_args()


def parse_csv(raw: str, *, allowed: set[str] | None = None) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one value is required")
    if allowed is not None:
        invalid = sorted(set(values) - allowed)
        if invalid:
            raise ValueError(f"unsupported values: {', '.join(invalid)}")
    return values


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    with resolved.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {resolved}")
    return payload


def make_run_config(base: dict[str, Any], stream_count: int) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    sources = list(config.get("sources", []))[:stream_count]
    if len(sources) != stream_count:
        raise ValueError(f"requested {stream_count} streams but config only has {len(sources)}")
    config.setdefault("app", {})["source_count"] = stream_count
    config["sources"] = sources
    # This is only the nvstreammux/primary-infer batch. Specialist workers
    # continue to run their own fixed batch-1 TensorRT engines.
    deepstream = config.setdefault("deepstream", {})
    deepstream["batch_size"] = stream_count
    rows, columns = {1: (1, 1), 4: (2, 2), 8: (2, 4)}.get(stream_count, (1, stream_count))
    deepstream["tiler_rows"] = rows
    deepstream["tiler_columns"] = columns
    return config


def override_primary_engine(config: dict[str, Any], engine_path: Path) -> dict[str, Any]:
    """Return a copied config with the same candidate path in both primary fields."""
    updated = json.loads(json.dumps(config))
    models = updated.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("primary"), dict):
        raise ValueError("INT8 configuration must define models.primary")
    engine = str(engine_path)
    models["primary"]["engine"] = engine
    updated.setdefault("deepstream", {})["model_engine_path"] = engine
    return updated


def validate_primary_model_ab(fp16: dict[str, Any], primary_int8: dict[str, Any]) -> None:
    """Reject a benchmark when any specialist model changes precision/path."""
    fp16_models = fp16.get("models", {})
    int8_models = primary_int8.get("models", {})
    if not isinstance(fp16_models, dict) or not isinstance(int8_models, dict):
        raise ValueError("both benchmark configurations must define models")
    for model_name in sorted(set(fp16_models) | set(int8_models)):
        if model_name == "primary":
            continue
        fp16_model = fp16_models.get(model_name, {})
        int8_model = int8_models.get(model_name, {})
        fp16_engine = fp16_model.get("engine") if isinstance(fp16_model, dict) else None
        int8_engine = int8_model.get("engine") if isinstance(int8_model, dict) else None
        if fp16_engine != int8_engine:
            raise ValueError(
                f"specialist model `{model_name}` differs between configurations: "
                f"{fp16_engine!r} != {int8_engine!r}"
            )
    fp16_primary = fp16_models.get("primary", {})
    int8_primary = int8_models.get("primary", {})
    fp16_engine = fp16_primary.get("engine") if isinstance(fp16_primary, dict) else None
    int8_engine = int8_primary.get("engine") if isinstance(int8_primary, dict) else None
    if not fp16_engine or not int8_engine or fp16_engine == int8_engine:
        raise ValueError("primary FP16 and INT8 engine paths must both exist and differ")
    if fp16.get("deepstream", {}).get("model_engine_path") != fp16_engine:
        raise ValueError("FP16 deepstream.model_engine_path must match models.primary.engine")
    if primary_int8.get("deepstream", {}).get("model_engine_path") != int8_engine:
        raise ValueError("INT8 deepstream.model_engine_path must match models.primary.engine")


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip()
    return text or None


def environment_snapshot() -> dict[str, str | None]:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "kernel": command_output(["uname", "-a"]),
        "jetson_release": command_output(["cat", "/etc/nv_tegra_release"]),
        "power_mode": command_output(["nvpmodel", "-q"]),
    }


def numeric(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["precision"], run["stream_count"], run["sink"])].append(run)
    summaries = []
    for (precision, stream_count, sink), group in sorted(groups.items()):
        successful = [item["summary"] for item in group if item.get("exit_code") == 0 and item.get("summary")]
        def avg(*keys: str) -> float | None:
            values = [numeric(item, *keys) for item in successful]
            values = [value for value in values if value is not None]
            return round(mean(values), 4) if values else None
        summaries.append({
            "precision": precision,
            "stream_count": stream_count,
            "sink": sink,
            "planned_repetitions": len(group),
            "successful_repetitions": len(successful),
            "average_processing_fps": avg("runtime", "average_processing_fps"),
            "average_end_to_end_p50_ms": avg("latency", "end_to_end_p50_ms"),
            "average_end_to_end_p95_ms": avg("latency", "end_to_end_p95_ms"),
            "average_gpu_utilization_percent": avg("runtime", "average_gpu_utilization_percent"),
            "average_ram_used_mb": avg("runtime", "average_ram_used_mb"),
            "average_gpu_soc_power_mw": avg("runtime", "average_gpu_soc_power_mw"),
            "average_temperature_c": avg("runtime", "average_temperature_c"),
        })
    return summaries


def main() -> int:
    args = parse_args()
    try:
        stream_counts = [int(item) for item in parse_csv(args.stream_counts)]
        sinks = parse_csv(args.sinks, allowed={"fake", "file"})
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if any(count <= 0 for count in stream_counts) or args.repetitions <= 0:
        raise SystemExit("stream counts and repetitions must be positive")
    for value, name in ((args.fp16_confidence_threshold, "--fp16-confidence-threshold"), (args.int8_confidence_threshold, "--int8-confidence-threshold")):
        if value is not None and not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be between 0 and 1")

    fp16 = load_config(args.fp16_config)
    primary_int8 = load_config(args.primary_int8_config)
    if args.primary_int8_engine is not None:
        primary_int8 = override_primary_engine(primary_int8, args.primary_int8_engine)
    validate_primary_model_ab(fp16, primary_int8)
    configs = {
        "primary_fp16": (fp16, args.fp16_confidence_threshold),
        "primary_int8": (primary_int8, args.int8_confidence_threshold),
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "environment.json").write_text(json.dumps(environment_snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    runs: list[dict[str, Any]] = []
    for precision, (base, confidence_threshold) in configs.items():
        for stream_count in stream_counts:
            generated = make_run_config(base, stream_count)
            for sink in sinks:
                for repetition in range(1, args.repetitions + 1):
                    label = f"{precision}_{stream_count}stream_{sink}_run{repetition}"
                    output_dir = root / label
                    output_dir.mkdir(parents=True, exist_ok=True)
                    config_path = output_dir / "config.yaml"
                    config_path.write_text(yaml.safe_dump(generated, allow_unicode=True, sort_keys=False), encoding="utf-8")
                    record: dict[str, Any] = {
                        "precision": precision,
                        "stream_count": stream_count,
                        "sink": sink,
                        "repetition": repetition,
                        "output_dir": str(output_dir),
                        "config": str(config_path),
                        "confidence_threshold": confidence_threshold,
                        "execute": args.execute,
                    }
                    if args.execute:
                        env = os.environ | {"OUTPUT_SINK": sink, "ENABLE_TEGRASTATS": "1", "RUN_SECONDS": str(args.run_seconds)}
                        if confidence_threshold is not None:
                            env["CONFIDENCE_THRESHOLD"] = str(confidence_threshold)
                        completed = subprocess.run(
                            ["scripts/deploy/run_multistream.sh", str(config_path), str(output_dir)],
                            cwd=PROJECT_ROOT,
                            env=env,
                            check=False,
                        )
                        record["exit_code"] = completed.returncode
                        if completed.returncode == 0:
                            record["summary"] = summarize(label, output_dir, warmup_samples=5)
                            (output_dir / "summary.json").write_text(
                                json.dumps(record["summary"], ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8",
                            )
                    runs.append(record)

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "executed": args.execute,
        "comparison": "primary_yolo_fp16_vs_int8_with_auxiliary_models_fp16",
        "primary_engines": {
            "fp16": fp16["models"]["primary"]["engine"],
            "int8": primary_int8["models"]["primary"]["engine"],
        },
        "confidence_thresholds": {"fp16": args.fp16_confidence_threshold, "int8": args.int8_confidence_threshold},
        "latency_definition": "primary_infer_sink_to_json_write_ms",
        "runs": runs,
        "aggregates": aggregate(runs) if args.execute else [],
    }
    report_path = root / "matrix_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark {'results' if args.execute else 'plan'}: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
