# PPE Micro-Batch Experiment

This experiment changes only the helmet/PPE worker. The primary detector stays
INT8 and all other specialist models remain FP16. It compares PPE batch sizes
1, 4, and 8 using the same eight local MP4 inputs and `fake` sink.

## Build Local Engines

The batch-1 engine is the existing deployment engine. Build separate local
engines for the two candidate profiles; engine files are ignored by Git.

```bash
scripts/benchmark/build_ppe_microbatch_engine.sh --max-batch 4
scripts/benchmark/build_ppe_microbatch_engine.sh --max-batch 8
```

## Run

First create a dry-run plan:

```bash
python3 scripts/benchmark/run_ppe_microbatch_matrix.py --repetitions 1
```

Then execute three repetitions per batch size:

```bash
python3 scripts/benchmark/run_ppe_microbatch_matrix.py --execute --repetitions 3
```

Results are written to `outputs/ppe_microbatch/<UTC timestamp>/`. Each run
contains its generated config, normal runtime artifacts, and `summary.json`.
The matrix `matrix_summary.json` contains aggregate throughput, PPE queue/task
latency, batch-size distribution, task/frame-store drops, and event signatures.

## Interpretation

A candidate is useful only when it reduces PPE task drops or raises PPE
processed count while keeping task P95 bounded and producing no unexplained
batch-1-only or candidate-only helmet events. It is normal for the average
batch to remain below the configured maximum when task arrivals are sparse.

This experiment does not prove model accuracy. It checks event consistency on
the same input workload; label-based accuracy still requires ground truth.

## 2026-08-11 Result and Deployment Decision

Result source: `outputs/ppe_microbatch/20260811T051103Z/matrix_summary.json`.
The matrix used the eight local MP4 sources, `fake` sink, primary INT8, all
other specialist engines FP16, and three end-of-stream repetitions per PPE
batch size.  The only intended change was the PPE worker batch/profile.

| PPE maximum batch | Actual average batch | System FPS | PPE P50 / P95 task latency (ms) | PPE processed | PPE task-buffer drops |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 72.96 | 334.74 / 870.84 | 424.33 | 793.00 |
| 4 | 3.16 | 69.13 | 732.16 / 1282.48 | 643.00 | 659.67 |
| 8 | 5.56 | 70.02 | 1195.93 / 1822.57 | 709.67 | 682.67 |

The actual batch distributions confirm that the dynamic engine and worker
aggregation paths were exercised (batch-4 median is 4; batch-8 P95 is 8).
Micro-batching reduced PPE task-buffer drops and increased PPE processed
requests, but it did **not** reduce real-time task latency: both P50 and P95
were worse than batch 1, while whole-system FPS was slightly lower.

Decision: the checked-in deployment configuration explicitly keeps PPE at
`micro_batch_size: 1` and `micro_batch_wait_ms: 0`. Batch 4/8 engines and the
matrix runner remain experimental tools, not production defaults. A future
trial may test batch 2 only if reducing PPE drops is more important than alert
latency; it must again report task P50/P95 and event behaviour.
