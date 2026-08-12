# Configuration Guide

Use one complete application configuration at a time; configuration fragments
are not merged automatically.

| Path | Purpose |
| --- | --- |
| `app/app_multifile_8.yaml` | Current 8-stream local-MP4 FP16 system baseline and default CLI configuration. |
| `app/app_multifile_8_primary_int8.yaml` | Current primary-detector INT8 candidate configuration; specialist models remain FP16. |
| `app/app_multifile_8_primary_int8_isolated_tasks.yaml` | Controlled scheduling experiment with per-task queues and stale-request deadlines. |
| `deepstream/infer_primary_yolo_minimal.txt` | Active primary nvinfer settings used by both supported application configurations. |
| `deepstream/streammux.yaml` | Shared streammux configuration. |
| `deepstream/tracker_iou.yml` | Shared NvMultiObjectTracker configuration. |
| `legacy/` | Earlier single-stream, person-only and six-stream settings. They are retained for traceability and are not default deployment inputs. |

Engine paths are local-only artifacts. Build the required FP16 or INT8 engine on
the target Jetson before running an application configuration.

`model_tasks.<name>.queue_size` is an independent latest-request queue capacity;
it does not share capacity with other tasks. `stale_after_ms` drops a request at
worker drain when its queue wait exceeds the configured deadline. These fields
are only set in the isolated-task experiment configuration so baseline results
remain comparable.

`optimization.frame_store_max_size` limits all retained source frames;
`frame_store_per_stream_capacity` additionally limits each stream. Leave both
unset for the historical derived capacity. The capacity matrix generates these
fields in runtime-only configs and does not change the checked-in baseline.
