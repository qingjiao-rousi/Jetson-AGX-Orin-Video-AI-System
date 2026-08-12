#include "probe_handler/probe_meta_parser.h"

// C++ 快路径只读取 probe 回调期间有效的 DeepStream metadata，并立即复制为普通值对象。
// 不把 NvDs* 指针交给 Python 或跨回调保存，避免 buffer 生命周期结束后的悬空引用。

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

#if __has_include("nvdsmeta.h")
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#define PROBE_HANDLER_HAS_DEEPSTREAM 1
#else
#define PROBE_HANDLER_HAS_DEEPSTREAM 0
#endif

namespace probe_handler {
namespace {

void copy_label(char* destination, const char* source) {
    // obj_label 属于 DeepStream metadata；复制到固定大小数组以让结果脱离原对象生命周期。
    if (destination == nullptr) {
        return;
    }
    const char* text = source == nullptr ? "unknown" : source;
    std::strncpy(destination, text, kMaxLabelLength - 1);
    destination[kMaxLabelLength - 1] = '\0';
}

std::string escape_json(const char* value) {
    // 不引入额外 JSON 依赖，输出仅包含标签等有限字符串字段，故在 ABI 边界手动转义。
    std::ostringstream out;
    const char* text = value == nullptr ? "" : value;
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        switch (*cursor) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << *cursor;
                break;
        }
    }
    return out.str();
}

}  // namespace

std::vector<ProbeFrameResult> parse_nvds_batch_meta(const _NvDsBatchMeta* batch_meta) {
    // 一个 GstBuffer 可被 nvstreammux 合成多帧，因此必须遍历 batch 内全部 frame_meta。
    std::vector<ProbeFrameResult> frames;
    if (batch_meta == nullptr) {
        return frames;
    }

#if PROBE_HANDLER_HAS_DEEPSTREAM
    const auto* nvds_batch_meta = reinterpret_cast<const NvDsBatchMeta*>(batch_meta);
    for (NvDsMetaList* frame_node = nvds_batch_meta->frame_meta_list; frame_node != nullptr;
         frame_node = frame_node->next) {
        const auto* frame_meta = reinterpret_cast<const NvDsFrameMeta*>(frame_node->data);
        if (frame_meta == nullptr) {
            continue;
        }

        ProbeFrameResult frame;
        // pad_index 是 streammux 输入序号，和 Python 的 canonical stream-N 对应。
        frame.stream_id = static_cast<uint32_t>(frame_meta->pad_index);
        frame.frame_id = static_cast<uint64_t>(frame_meta->frame_num);
        frame.ntp_timestamp = static_cast<uint64_t>(frame_meta->ntp_timestamp);

        for (NvDsMetaList* obj_node = frame_meta->obj_meta_list; obj_node != nullptr;
             obj_node = obj_node->next) {
            const auto* obj_meta = reinterpret_cast<const NvDsObjectMeta*>(obj_node->data);
            if (obj_meta == nullptr) {
                continue;
            }

            ProbeDetection detection;
            detection.class_id = static_cast<int32_t>(obj_meta->class_id);
            detection.track_id = static_cast<uint64_t>(obj_meta->object_id);
            detection.confidence = static_cast<float>(obj_meta->confidence);
            detection.bbox.left = static_cast<float>(obj_meta->rect_params.left);
            detection.bbox.top = static_cast<float>(obj_meta->rect_params.top);
            detection.bbox.width = static_cast<float>(obj_meta->rect_params.width);
            detection.bbox.height = static_cast<float>(obj_meta->rect_params.height);
            copy_label(detection.class_name, obj_meta->obj_label);
            frame.detections.push_back(detection);
        }

        frames.push_back(std::move(frame));
    }
#endif

    return frames;
}

std::string frames_to_json(const std::vector<ProbeFrameResult>& frames) {
    // JSON 不是最终业务格式，而是稳定且易于 ctypes 传递的 C++/Python ABI 中间表示。
    std::ostringstream out;
    out << '[';
    for (std::size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
        const auto& frame = frames[frame_index];
        if (frame_index > 0) {
            out << ',';
        }
        out << "{\"stream_id\":\"stream-" << frame.stream_id << "\",";
        out << "\"source_id\":" << frame.stream_id << ',';
        out << "\"frame_id\":" << frame.frame_id << ',';
        out << "\"ntp_timestamp\":" << frame.ntp_timestamp << ',';
        out << "\"detections\":[";

        for (std::size_t detection_index = 0; detection_index < frame.detections.size(); ++detection_index) {
            const auto& detection = frame.detections[detection_index];
            if (detection_index > 0) {
                out << ',';
            }
            out << "{\"class_id\":" << detection.class_id << ',';
            out << "\"class_name\":\"" << escape_json(detection.class_name) << "\",";
            out << "\"confidence\":" << std::fixed << std::setprecision(6) << detection.confidence << ',';
            out << "\"track_id\":" << detection.track_id << ',';
            out << "\"bbox\":{";
            out << "\"left\":" << detection.bbox.left << ',';
            out << "\"top\":" << detection.bbox.top << ',';
            out << "\"width\":" << detection.bbox.width << ',';
            out << "\"height\":" << detection.bbox.height << "}}";
        }

        out << "]}";
    }
    out << ']';
    return out.str();
}

}  // namespace probe_handler

extern "C" {

namespace {

const char* allocate_json(const _NvDsBatchMeta* batch_meta) {
    // 分配所有权明确交给 probe_free_json，不能返回 std::string::c_str() 的临时指针。
    const auto frames = probe_handler::parse_nvds_batch_meta(batch_meta);
    const auto payload = probe_handler::frames_to_json(frames);
    auto* result = new char[payload.size() + 1];
    std::memcpy(result, payload.c_str(), payload.size() + 1);
    return result;
}

}  // namespace

const char* probe_parse_nvds_batch_meta_json(const _NvDsBatchMeta* batch_meta) {
    return allocate_json(batch_meta);
}

const char* probe_parse_gst_buffer_json(const void* buffer) {
#if PROBE_HANDLER_HAS_DEEPSTREAM
    if (buffer == nullptr) {
        return allocate_json(nullptr);
    }
    // Python ctypes 传入的是 GstBuffer 地址；仅在当前 probe 回调内读取其 batch metadata。
    auto* gst_buffer = reinterpret_cast<GstBuffer*>(const_cast<void*>(buffer));
    return allocate_json(reinterpret_cast<const _NvDsBatchMeta*>(
        gst_buffer_get_nvds_batch_meta(gst_buffer)));
#else
    (void)buffer;
    return allocate_json(nullptr);
#endif
}

void probe_free_json(const char* payload) {
    delete[] payload;
}

}
