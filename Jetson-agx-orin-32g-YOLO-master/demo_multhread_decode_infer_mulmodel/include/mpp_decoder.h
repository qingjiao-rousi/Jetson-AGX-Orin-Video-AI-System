/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#pragma once
#include <cstring>
#include "mpp_frame.h"
#include "rk_mpi.h"
#include <functional>
#include <mutex>
#define FRAME_SYNC_TIME 0
// 定义解码完成后回调函数的格式，用于在解码完成后调用
using MppDecoderFrameCallback = std::function<void(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data, int id)>;

typedef struct
{
    MppCtx ctx;             // MPP上下文,一个硬件编解码器实例
    MppApi *mpi;            // MPP API接口,用于调用MPP API函数
    RK_U32 eos;             // 是否是结束标志
    MppBufferGroup frm_grp; // 图像数据缓冲区组，用于存储解码后的图像数据
    MppBufferGroup pkt_grp; // 数据包缓冲区组，用于存储解码后的数据包
    MppPacket packet;       // 对应从AVPacket转换过来的MPP包，里面放的是H.264/H.265压缩码流
    MppFrame frame;         // MPP解码出来的一帧图像
    size_t max_usage;       // 最大内存占用量
} MpiDecLoopData;           // MPP解码器循环数据结构体，用于存储解码器的循环数据

class MppDecoder
{
public:
    MppCtx mpp_ctx = NULL;
    MppApi *mpp_mpi = NULL;
    MppDecoder();
    ~MppDecoder();
    int Init(int video_type, int fps, void *userdata, int id); // 初始化解码器
    int SetCallback(MppDecoderFrameCallback callback);         // 设置解码完成后回调函数
    int Decode(uint8_t *pkt_data, int pkt_size, int pkt_eos);  // 解码一帧视频
    int Reset();

private:
    MppParam mpp_param1 = NULL;
    RK_U32 need_split = 1; // 开启MPP内部码流切分功能
    RK_U32 width_mpp;
    RK_U32 height_mpp;
    MppCodingType mpp_type; // MPP编码类型，H.264/H.265
    size_t packet_size = 2400 * 1300 * 3 / 2;
    MpiDecLoopData loop_data; // MPP解码器循环数据结构体，用于存储解码器的循环数据
    MppPacket packet = NULL;
    MppFrame frame = NULL;
    MppDecoderFrameCallback callback;
    int fps = -1;
    unsigned long last_frame_time_ms = 0;
    void *userdata = NULL; // 回调时传递的用户数据
    int id = 0;
    std::mutex mtx;
};
