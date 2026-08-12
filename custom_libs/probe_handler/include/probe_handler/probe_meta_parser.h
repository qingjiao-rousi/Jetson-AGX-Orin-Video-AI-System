#pragma once

// DeepStream probe 的原生快路径接口。
// 此头文件刻意只前置声明 NvDsBatchMeta，避免公共接口强耦合 DeepStream 头文件。

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct _NvDsBatchMeta;

namespace probe_handler {

constexpr std::size_t kMaxLabelLength = 64;

struct ProbeBoundingBox {
    // 保持 DeepStream rect_params 的 left/top/width/height 语义，不转换为 xyxy。
    float left = 0.0F;
    float top = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
};

struct ProbeDetection {
    // track_id 是 DeepStream tracker 的原始 object_id；本地连续 ID 在 Python 层归一化。
    int32_t class_id = 0;
    uint64_t track_id = 0;
    float confidence = 0.0F;
    ProbeBoundingBox bbox;
    char class_name[kMaxLabelLength] = {};
};

struct ProbeFrameResult {
    // stream_id 对应 nvstreammux pad_index，最终被序列化为 stream-N。
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
const char* probe_parse_gst_buffer_json(const void* buffer);
// parse 函数返回由 new[] 分配的 UTF-8 JSON；调用方必须恰好调用一次此函数释放。
void probe_free_json(const char* payload);

}
