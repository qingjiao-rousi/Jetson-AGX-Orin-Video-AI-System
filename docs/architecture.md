# System Architecture

The project uses one DeepStream pipeline for multi-stream primary detection and
keeps scenario-specific inference outside the GStreamer probe callback.

```text
Local MP4 or RTSP sources
  -> hardware decode
  -> nvstreammux (batch 1, 4, or 8)
  -> primary TensorRT YOLO + tracker + OSD
  -> C++ NvDsBatchMeta parser
  -> Python FrameResult and routing policy
  -> bounded specialist workers
  -> JSONL events, runtime metrics, and selected video sink
```

## Boundaries

- `src/app/infrastructure/pipeline/` owns GStreamer and DeepStream lifecycle.
- `custom_libs/probe_handler/` extracts batch metadata in C++ and has a Python
  fallback path for non-target environments.
- `src/app/application/` owns scene routing, task queues, specialist workers,
  and event state.
- `src/app/infrastructure/output/` owns asynchronous JSONL output.
- `src/app/infrastructure/monitoring/` records `tegrastats` and queue metrics.

The primary detector is batch-capable and is the only precision variable in
the documented FP16/INT8 benchmark. Specialist workers currently submit one
request at a time; safety helmet, pose, and fire/smoke micro-batching are
future optimization work. Plate detection and OCR remain outside that scope.
