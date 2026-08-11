from __future__ import annotations

import unittest
from threading import Event, Thread

from app.application.task_metrics import TaskExecutionMetrics


class TaskExecutionMetricsTests(unittest.TestCase):
    def test_exposes_bounded_execution_summaries(self) -> None:
        metrics = TaskExecutionMetrics(sample_limit=2)
        metrics.processed = 3
        metrics.missing_frames = 1
        metrics.errors = 2
        metrics.record_queue_wait(10)
        metrics.record_queue_wait(20)
        metrics.record_queue_wait(30)
        metrics.record_inference(4)
        metrics.record_task_latency(40)
        stats = metrics.stats()
        self.assertEqual(stats["processed"], 3)
        self.assertEqual(stats["queue_wait_ms"]["samples"], 2)
        self.assertEqual(stats["queue_wait_ms"]["p50"], 25.0)
        self.assertEqual(stats["inference_ms"]["p95"], 4.0)

    def test_stats_is_safe_while_worker_records_samples(self) -> None:
        metrics = TaskExecutionMetrics()
        started = Event()
        stop = Event()

        def writer() -> None:
            started.set()
            value = 0
            while not stop.is_set():
                metrics.record_queue_wait(value)
                value += 1

        thread = Thread(target=writer)
        thread.start()
        started.wait(timeout=1)
        try:
            for _ in range(100):
                metrics.stats()
        finally:
            stop.set()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
