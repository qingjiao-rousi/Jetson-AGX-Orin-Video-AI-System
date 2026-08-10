from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class OptimizationAdvisor:
    """
    Runtime-aware optimization advisor.

    It reports the controls that are active and leaves policy decisions to the
    FPS gate and backpressure controller.
    """

    def recommend(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        actions: list[dict[str, str]] = []

        pipeline_status = snapshot.get("pipeline_status", {})
        bus = snapshot.get("bus", {})
        controllers = snapshot.get("controllers", {})
        monitor = snapshot.get("monitor", {})

        if snapshot.get("last_error"):
            actions.append(
                {
                    "priority": "high",
                    "action": "stabilize_pipeline",
                    "reason": "Pipeline has a recorded error and should be stabilized before performance tuning.",
                }
            )

        if bus.get("last_warning"):
            actions.append(
                {
                    "priority": "medium",
                    "action": "inspect_bus_warning",
                    "reason": "A recent GStreamer/DeepStream warning should be reviewed before scaling routes.",
                }
            )

        if pipeline_status.get("pipeline_state") == "PLAYING":
            actions.append(
                {
                    "priority": "medium",
                    "action": "prepare_real_metrics",
                    "reason": "The pipeline is running, so real FPS, latency, and queue metrics can be wired next.",
                }
            )

        if controllers.get("backpressure", {}).get("queue_limit") is not None:
            actions.append(
                {
                    "priority": "medium",
                    "action": "monitor_backpressure_policy",
                    "reason": "Backpressure is connected to the asynchronous result writer completion signal.",
                }
            )

        if not monitor or monitor.get("status") in {None, "started"}:
            actions.append(
                {
                    "priority": "low",
                    "action": "enrich_gpu_monitor",
                    "reason": "GPU monitoring can be extended from placeholder status to real Jetson metrics.",
                }
            )

        actions.extend(
            [
                {
                    "priority": "low",
                    "action": "fp16_to_int8",
                    "reason": "INT8 engines are available; compare them with a rerun FP16 baseline on the same input and metrics contract.",
                },
                {
                    "priority": "low",
                    "action": "result_async_output",
                    "reason": "Structured result output can later be decoupled with an async consumer queue if JSONL or MQTT becomes heavy.",
                },
            ]
        )

        return {
            "mode": "runtime_controls_with_advisory_recommendations",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "action_count": len(actions),
            "actions": actions,
        }
