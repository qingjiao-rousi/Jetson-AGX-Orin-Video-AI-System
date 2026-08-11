# Script Guide

Scripts are grouped by operational responsibility. Run every command from the
repository root so relative model and configuration paths resolve consistently.

| Directory | Purpose | Primary entry points |
| --- | --- | --- |
| `deploy/` | Jetson environment, parser and engine builds, application launch, systemd installation | `env.sh`, `check_env.sh`, `build_fp16_engines.sh`, `build_primary_detector_int8.py`, `run_multistream.sh` |
| `benchmark/` | Repeatable system and PPE micro-batch benchmarks | `run_benchmark_matrix.py`, `run_ppe_microbatch_matrix.py`, `summarize_precision_run.py` |
| `evaluation/` | Offline FP16/INT8 output comparison, COCO evaluation and calibration manifests | `align_primary_detector_outputs.py`, `evaluate_primary_detector_coco.py`, `prepare_coco_train_calibration.py` |
| `rtsp/` | Local RTSP simulation, RTSP pipeline runs, recovery checks and acceptance flows | `simulate_cameras.sh`, `run_rtsp_inproc.sh`, `run_rtsp_acceptance.sh` |
| `tools/` | Optional local utilities, including ONNX export, Q/DQ experiments and the preview server | `export_pt_to_onnx.py`, `quantize_yolov8s_qdq.py`, `preview_web.py` |
| `legacy/person_analytics/` | Earlier person-only analytics and acceptance workflow, retained for traceability but not the default deployment path | `run_person_analytics.sh`, `run_multifile_inproc.sh` |

The repository does not include engines, ONNX files, calibration images, videos
or output artifacts. See `models/README.md` before running deployment scripts.
