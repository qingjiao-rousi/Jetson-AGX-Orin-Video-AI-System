#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct _NvDsBatchMeta;

namespace probe_handler {

constexpr std::size_t kMaxLabelLength = 64;

struct ProbeBoundingBox {
    float left = 0.0F;
    float top = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
};

struct ProbeDetection {
    int32_t class_id = 0;
    uint64_t track_id = 0;
    float confidence = 0.0F;
    ProbeBoundingBox bbox;
    char class_name[kMaxLabelLength] = {};
};

struct ProbeFrameResult {
    uint32_t stream_id = 0;
    uint64_t frame_id = 0;
    uint64_t ntp_timestamp = 0;
    std::vector<ProbeDetection> detections;
};

std::vector<ProbeFrameResult> parse_nvds_batch_meta(const _NvDsBatchMeta* batch_meta);
std::string frames_to_json(const std::vector<ProbeFrameResult>& frames);

}  // namespace probe_handler

extern "C" {

using ProbeJsonFreeFn = void (*)(const char*);

const char* probe_parse_nvds_batch_meta_json(const _NvDsBatchMeta* batch_meta);
void probe_free_json(const char* payload);

}
