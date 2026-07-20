from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.settings import AppSettings, SourceSettings


PERSON_FILTER_OUT_CLASS_IDS = ";".join(str(class_id) for class_id in range(1, 80))


def apply_runtime_overrides(
    settings: AppSettings,
    *,
    input_video: Path | None = None,
    output_video: Path | None = None,
    output_json: Path | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    confidence_threshold: float | None = None,
    person_only: bool = True,
    enable_web: bool | None = None,
    runtime_dir: Path | None = None,
    output_dir: Path | None = None,
    output_sink: str | None = None,
    output_url: str | None = None,
) -> AppSettings:
    deepstream = settings.deepstream
    output = settings.output
    sources = settings.sources
    source_count = settings.source_count
    web = settings.web

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = replace(
            output,
            jsonl_path=output_dir / "results.jsonl",
            events_jsonl_path=output_dir / "events.jsonl",
            metrics_jsonl_path=output_dir / "runtime_metrics.jsonl",
        )
        deepstream = replace(
            deepstream,
            output_video_path=output_dir / "output.mp4",
        )
        settings_logging = replace(settings.logging, file_path=output_dir / "app.log")
        web = replace(web, batch_dir=output_dir, multifile_dir=output_dir, rtsp_dir=output_dir)
    else:
        settings_logging = settings.logging

    if enable_web is not None:
        web = replace(web, enabled=enable_web)

    if input_video is not None:
        sources = (
            SourceSettings(
                name="local_video_01",
                uri=str(input_video),
                kind="file",
                enabled=True,
            ),
        )
        source_count = 1
        deepstream = replace(deepstream, batch_size=1)

    if output_video is not None:
        deepstream = replace(
            deepstream,
            output_sink="file",
            output_video_path=output_video,
        )

    if output_sink is not None or output_url is not None:
        deepstream = replace(
            deepstream,
            output_sink=output_sink or deepstream.output_sink,
            output_url=output_url or deepstream.output_url,
        )

    if output_json is not None:
        output = replace(output, jsonl_path=output_json, enable_jsonl=True)

    if output_width is not None or output_height is not None:
        deepstream = replace(
            deepstream,
            inference_width=output_width or deepstream.inference_width,
            inference_height=output_height or deepstream.inference_height,
        )

    if confidence_threshold is not None or person_only is not None:
        runtime_infer_config = _write_runtime_infer_config(
            deepstream.infer_config_path,
            confidence_threshold=confidence_threshold,
            person_only=person_only,
            batch_size=deepstream.batch_size,
            infer_interval=deepstream.infer_interval,
            runtime_dir=runtime_dir,
        )
        deepstream = replace(deepstream, infer_config_path=runtime_infer_config)

    return replace(
        settings,
        source_count=source_count,
        sources=sources,
        output=output,
        logging=settings_logging,
        enable_web=web.enabled,
        web=web,
        deepstream=deepstream,
    )


def _write_runtime_infer_config(
    base_path: Path,
    *,
    confidence_threshold: float | None,
    person_only: bool,
    batch_size: int,
    infer_interval: int,
    runtime_dir: Path | None = None,
) -> Path:
    text = base_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    updated: list[str] = []
    saw_threshold = False
    saw_filter = False
    saw_interval = False

    for line in lines:
        stripped = line.strip()
        if _is_infer_path_property(stripped):
            key, raw_value = stripped.split("=", 1)
            updated.append(f"{key}={_resolve_runtime_config_path(raw_value)}")
            continue

        if stripped.startswith("batch-size="):
            updated.append(f"batch-size={max(int(batch_size), 1)}")
            continue

        if stripped.startswith("interval="):
            saw_interval = True
            updated.append(f"interval={max(int(infer_interval), 0)}")
            continue

        if stripped.startswith("pre-cluster-threshold="):
            saw_threshold = True
            if confidence_threshold is not None:
                updated.append(f"pre-cluster-threshold={confidence_threshold:.4f}")
            else:
                updated.append(line)
            continue

        if stripped.startswith("filter-out-class-ids=") or stripped.startswith("# filter-out-class-ids="):
            saw_filter = True
            if person_only:
                updated.append(f"filter-out-class-ids={PERSON_FILTER_OUT_CLASS_IDS}")
            else:
                updated.append(f"# filter-out-class-ids={PERSON_FILTER_OUT_CLASS_IDS}")
            continue

        updated.append(line)

    if not saw_filter and person_only:
        insert_at = _find_property_insert_index(updated)
        updated.insert(insert_at, f"filter-out-class-ids={PERSON_FILTER_OUT_CLASS_IDS}")

    if not saw_interval:
        insert_at = _find_property_insert_index(updated)
        updated.insert(insert_at, f"interval={max(int(infer_interval), 0)}")

    if confidence_threshold is not None and not saw_threshold:
        updated.extend(["", "[class-attrs-all]", f"pre-cluster-threshold={confidence_threshold:.4f}"])

    runtime_dir = runtime_dir or Path("outputs/runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / base_path.name
    runtime_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return runtime_path


def _is_infer_path_property(line: str) -> bool:
    if "=" not in line or line.startswith("#"):
        return False
    key, _value = line.split("=", 1)
    return key in {
        "model-engine-file",
        "onnx-file",
        "labelfile-path",
        "custom-lib-path",
    }


def _resolve_runtime_config_path(raw_value: str) -> str:
    path = Path(raw_value.strip())
    if path.is_absolute():
        return str(path)
    return str(path.resolve())


def _find_property_insert_index(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.strip().startswith("[class-attrs-"):
            return index
    return len(lines)
