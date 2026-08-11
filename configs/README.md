# Configuration Guide

Use one complete application configuration at a time; configuration fragments
are not merged automatically.

| Path | Purpose |
| --- | --- |
| `app/app_multifile_8.yaml` | Current 8-stream local-MP4 FP16 system baseline and default CLI configuration. |
| `app/app_multifile_8_primary_int8.yaml` | Current primary-detector INT8 candidate configuration; specialist models remain FP16. |
| `deepstream/infer_primary_yolo_minimal.txt` | Active primary nvinfer settings used by both supported application configurations. |
| `deepstream/streammux.yaml` | Shared streammux configuration. |
| `deepstream/tracker_iou.yml` | Shared NvMultiObjectTracker configuration. |
| `legacy/` | Earlier single-stream, person-only and six-stream settings. They are retained for traceability and are not default deployment inputs. |

Engine paths are local-only artifacts. Build the required FP16 or INT8 engine on
the target Jetson before running an application configuration.
