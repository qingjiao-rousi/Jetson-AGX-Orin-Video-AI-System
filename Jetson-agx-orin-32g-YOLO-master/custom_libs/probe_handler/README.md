# C++ Probe / Meta Parser

This module moves the hot-path `NvDsBatchMeta` traversal out of Python and into C++.

The first version exposes a small C ABI:

- `probe_parse_nvds_batch_meta_json(const _NvDsBatchMeta*)`
- `probe_free_json(const char*)`

On Jetson with DeepStream headers available, `src/probe_meta_parser.cpp` includes `nvdsmeta.h` and extracts:

- stream id
- frame id
- NTP timestamp
- class id / label / confidence
- bounding box
- tracker object id

On a development machine without DeepStream, the library still compiles through the offline stub path so the project structure and ABI can be checked before board validation.

Build on Jetson:

```bash
cmake -S custom_libs/probe_handler -B build/probe_handler
cmake --build build/probe_handler -j
```
