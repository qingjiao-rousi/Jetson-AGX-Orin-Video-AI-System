from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time


class GpuMonitor:
    def __init__(self, *, interval_ms: int | None = None, enabled: bool | None = None) -> None:
        env_enabled = os.environ.get("ENABLE_TEGRASTATS", "1") != "0"
        self._enabled = env_enabled if enabled is None else enabled
        self._interval_ms = interval_ms or int(os.environ.get("TEGRASTATS_INTERVAL_MS", "1000"))
        self._running = False
        self._last_snapshot: dict[str, object] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._set_snapshot(
            {
                "status": "starting",
                "running": True,
                "provider": "tegrastats",
                "utilization_gpu": None,
                "utilization_memory": None,
                "temperature_c": None,
            }
        )
        if not self._enabled:
            self._set_unavailable("disabled by ENABLE_TEGRASTATS=0")
            return
        binary = shutil.which("tegrastats")
        if not binary:
            self._set_unavailable("tegrastats binary not found")
            return
        try:
            self._process = subprocess.Popen(
                [binary, "--interval", str(max(self._interval_ms, 100))],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._set_unavailable(f"failed to start tegrastats: {exc}")
            return
        self._thread = threading.Thread(target=self._read_loop, name="tegrastats-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        self._thread = None
        snapshot = self.snapshot() or {}
        snapshot.update(
            {
                "status": "stopped",
                "running": False,
            }
        )
        self._set_snapshot(snapshot)

    @property
    def running(self) -> bool:
        return self._running

    def gpu_util(self) -> float | None:
        snapshot = self.snapshot() or {}
        value = snapshot.get("utilization_gpu")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def snapshot(self) -> dict[str, object] | None:
        with self._lock:
            return dict(self._last_snapshot) if self._last_snapshot is not None else None

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        while self._running:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    self._set_unavailable(f"tegrastats exited with code {process.returncode}")
                    return
                time.sleep(0.1)
                continue
            try:
                snapshot = _parse_tegrastats_line(line)
                snapshot.update(
                    {
                        "status": "ok",
                        "running": True,
                        "provider": "tegrastats",
                        "raw": line.strip(),
                    }
                )
                self._set_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - defensive monitor path
                logging.debug("failed to parse tegrastats line: %s", exc)

    def _set_unavailable(self, reason: str) -> None:
        self._set_snapshot(
            {
                "status": "unavailable",
                "running": self._running,
                "provider": "tegrastats",
                "utilization_gpu": None,
                "utilization_memory": None,
                "temperature_c": None,
                "reason": reason,
            }
        )

    def _set_snapshot(self, snapshot: dict[str, object]) -> None:
        with self._lock:
            self._last_snapshot = snapshot


def _parse_tegrastats_line(line: str) -> dict[str, object]:
    ram_match = re.search(r"\bRAM\s+(\d+)/(\d+)MB", line)
    swap_match = re.search(r"\bSWAP\s+(\d+)/(\d+)MB", line)
    gr3d_match = re.search(r"\bGR3D_FREQ\s+(\d+)%", line)
    emc_match = re.search(r"\bEMC_FREQ\s+(\d+)%", line)
    power_match = re.search(r"\bVDD_GPU_SOC\s+(\d+)mW/(\d+)mW", line)
    if power_match is None:
        power_match = re.search(r"\bVDD_GPU\s+(\d+)mW/(\d+)mW", line)
    temps = {
        key.lower(): float(value)
        for key, value in re.findall(r"\b([A-Za-z0-9_]+)@([0-9.]+)C", line)
    }

    ram_used = _int_group(ram_match, 1)
    ram_total = _int_group(ram_match, 2)
    memory_pct = round((ram_used / ram_total) * 100, 1) if ram_used is not None and ram_total else None
    gpu_temp = temps.get("gpu") or temps.get("gpu0") or temps.get("soc2")

    return {
        "utilization_gpu": _int_group(gr3d_match, 1),
        "utilization_memory": memory_pct,
        "temperature_c": gpu_temp,
        "ram_used_mb": ram_used,
        "ram_total_mb": ram_total,
        "swap_used_mb": _int_group(swap_match, 1),
        "swap_total_mb": _int_group(swap_match, 2),
        "emc_utilization": _int_group(emc_match, 1),
        "power_gpu_soc_mw": _int_group(power_match, 1),
        "power_gpu_soc_avg_mw": _int_group(power_match, 2),
        "temperatures_c": temps,
    }


def _int_group(match: re.Match[str] | None, index: int) -> int | None:
    if match is None:
        return None
    try:
        return int(match.group(index))
    except (TypeError, ValueError):
        return None
