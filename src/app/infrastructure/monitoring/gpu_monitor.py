from __future__ import annotations


class GpuMonitor:
    def __init__(self) -> None:
        self._running = False
        self._last_snapshot: dict[str, object] | None = None

    def start(self) -> None:
        self._running = True
        self._last_snapshot = {
            "status": "started",
            "running": True,
            "utilization_gpu": None,
            "utilization_memory": None,
            "temperature_c": None,
            "note": "placeholder monitor until Jetson runtime metrics are connected",
        }

    def stop(self) -> None:
        self._running = False
        self._last_snapshot = {
            "status": "stopped",
            "running": False,
            "utilization_gpu": None,
            "utilization_memory": None,
            "temperature_c": None,
        }

    @property
    def running(self) -> bool:
        return self._running

    def snapshot(self) -> dict[str, object] | None:
        return self._last_snapshot
