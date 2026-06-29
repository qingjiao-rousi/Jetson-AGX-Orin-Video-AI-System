from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class DebugService:
    orchestrator: object
    log_buffer: object
    optimization_advisor: object

    def health_snapshot(self) -> dict[str, Any]:
        status = self.orchestrator.status_snapshot()
        pipeline_status = status.get("pipeline_status", {})
        return {
            "app_name": getattr(getattr(self.orchestrator, "settings", None), "app_name", None),
            "healthy": status.get("last_error") is None,
            "is_running": status.get("is_running", False),
            "pipeline_state": pipeline_status.get("pipeline_state", "UNKNOWN"),
            "generated_at": datetime.now(UTC).isoformat(),
        }

    def status_snapshot(self) -> dict[str, Any]:
        status = self.orchestrator.status_snapshot()
        return {
            **status,
            "logs": self.log_buffer.stats() if hasattr(self.log_buffer, "stats") else {},
        }

    def logs_snapshot(self, limit: int = 100) -> dict[str, Any]:
        safe_limit = max(1, min(limit, 1000))
        return {
            "limit": safe_limit,
            "items": self.log_buffer.tail(safe_limit) if hasattr(self.log_buffer, "tail") else [],
        }

    def debug_snapshot(self, limit: int = 100) -> dict[str, Any]:
        status = self.status_snapshot()
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "health": self.health_snapshot(),
            "status": status,
            "optimization": self.optimization_advisor.recommend(status)
            if hasattr(self.optimization_advisor, "recommend")
            else {},
            "recent_logs": self.logs_snapshot(limit),
        }
