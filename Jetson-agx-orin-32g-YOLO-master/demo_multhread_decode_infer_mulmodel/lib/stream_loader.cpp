/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#include "stream_loader.h"
#include "im2d.h"
#include <chrono>
#include <string>
#include <thread>

// 判断是否为 Annex B 格式
// 该函数并没有使用
int is_annexb(const uint8_t *buf, size_t buf_size) // 判断一段 H.264 数据是否是 Annex B 格式。
{
    // Annex B 格式以 0x000001 或 0x00000001 开头
    if (buf_size >= 4)
    {
        if ((buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x01) ||
            (buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x00 && buf[3] == 0x01))
        {
            return 1; // 是 Annex B 格式
        }
    }
    return 0; // 不是 Annex B 格式
} // 要判断是否是 Annex B 格式的 H.264 数据

// MppDecoder 解码出一帧后调用
void mpp_decoder_frame_callback(void *buffer, int width_stride, int height_stride, int width, int height, int format, int fd, void *data, int id)
{
    Mbuffer *mbuffer = (Mbuffer *)buffer;             // 其中buffer是void*类型，需要强制类型转换为Mbuffer*类型
    size_t yuv_size = (size_t)width * height * 3 / 2; // 计算 NV12 格式的图像数据大小

    mbuffer->yuv_work.resize(yuv_size);           // 给yuv_work分配内存
    uint8_t *yuv_data = mbuffer->yuv_work.data(); // 获取yuv_work的指针

    uint8_t *base_y = (uint8_t *)data;                               // data 是硬件解码器输出的一帧图像数据的起始指针。
    uint8_t *base_c = base_y + (size_t)width_stride * height_stride; // uv数据的起始指针
    int idx = 0;                                                     // yuv_data的索引

    for (int i = 0; i < height; i++, base_y += width_stride) // 复制 Y 数据
    {
        memcpy(yuv_data + idx, base_y, width);
        idx += width;
    }

    for (int i = 0; i < height / 2; i++, base_c += width_stride) // 复制 U 数据
    {
        memcpy(yuv_data + idx, base_c, width); // 复制 U 数据
        idx += width;
    }

    mbuffer->bgr_work.create(height, width, CV_8UC3); // 创建BGR格式缓冲区
    if (mbuffer->bgr_work.empty())                    // 检查BGR格式缓冲区是否为空
    {
        return;
    }

    // 优先使用 RGA 硬件加速 NV12->BGR，失败则回退 OpenCV
    // 将 YUV 数据包装成 RGA 库能识别的源图像格式。
    rga_buffer_t src_buf = wrapbuffer_virtualaddr_t(yuv_data, width, height, width, height, RK_FORMAT_YCbCr_420_SP); // RK_FORMAT_YCbCr_420_SP	NV12 格式（YUV420 半平面）
    rga_buffer_t dst_buf = wrapbuffer_virtualaddr_t(mbuffer->bgr_work.data, width, height, width, height, RK_FORMAT_BGR_888);

    IM_STATUS status = imcvtcolor(src_buf, dst_buf,
                                  RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888, // 执行 RGA 格式转换
                                  IM_COLOR_SPACE_DEFAULT);

    if (status != IM_STATUS_SUCCESS) // RGA 格式转换失败
    {
        static int fallback_count = 0; // 防止大量重复错误刷屏,同时避免第一次出现问题，留有回复的机会
        if (fallback_count++ < 3)      // RGA 格式转换失败次数小于3次
        {
            fprintf(stderr, "RGA NV12->BGR failed (%d), using OpenCV fallback\n", (int)status);
        }
        cv::Mat yuvMat(height + height / 2, width, CV_8UC1, yuv_data);   // 包装 YUV 数据为 OpenCV Mat
        cv::cvtColor(yuvMat, mbuffer->bgr_work, cv::COLOR_YUV2BGR_NV12); // 将 NV12 格式的 YUV 图像转换为 BGR 彩色图像
    }

    std::unique_lock<std::mutex> mlock(mbuffer->mtx); // mbuffer->img 可能被其他线程（如显示线程、编码线程）读取
    mbuffer->img = std::move(mbuffer->bgr_work);      // 图像数据转移给 img
    mlock.unlock();

    // 每输出一帧限速，解决一包多帧导致的倍速
    if (mbuffer->throttle && mbuffer->frame_interval_ms > 0)                                // 节流、限速
        std::this_thread::sleep_for(std::chrono::milliseconds(mbuffer->frame_interval_ms)); // 控制每帧进入共享缓冲区的速度，避免下游挤压
}

void StreamLoader::close() // 清理 FFmpeg 解码器使用的各种资源
{
    decoder.Reset();
    if (temp_pkt)
    {
        av_packet_free(&temp_pkt); // 释放 temp_pkt 并将指针置为 nullptr
    }

    if (fmtCtx)
    {
        avformat_close_input(&fmtCtx); // 关闭输入流
        fmtCtx = nullptr;              // 确保指针在关闭后被设置为 nullptr
    }

    if (codecPar)
    {
        avcodec_parameters_free(&codecPar); // 释放 codecPar 结构
    }
}

bool StreamLoader::read_frame() // 从视频文件中读取数据包，解码，并返回是否成功获得一帧图像。
{
    using namespace std::chrono_literals;
    int eof_retry = 0;                    // 连续 av_read_frame 失败次数（EOF 时递增）
    int no_frame_count = 0;               // 已读视频包但解码未出帧的次数，防止异常时死循环
    const int MAX_EOF_RETRY = 10;         // EOF 时最多重试次数，超过则触发 reconnect
    const int MAX_PACKETS_NO_FRAME = 100; // 连续读包未出帧的上限，避免异常流导致死循环

    while (true)
    {
        int x = av_read_frame(fmtCtx, temp_pkt); //>= 0 成功读到包；<0 读取失败
        if (x < 0)
        {
            status = x;
            eof_retry++;
            if (eof_retry >= MAX_EOF_RETRY)
            {
                return false; // 确认 EOF，触发 reconnect 循环播放
            }
            std::this_thread::sleep_for(2ms); // 重试间隔 2ms
            av_packet_unref(temp_pkt);        // 释放 temp_pkt 并将指针置为 nullptr，为下一次读取做准备
            continue;
        }

        eof_retry = 0; // 成功读到包，重置 EOF 计数

        if (temp_pkt->stream_index != videoStreamIndex) // 过滤非视频包
        {
            av_packet_unref(temp_pkt);
            continue;
        }

        // 视频包
        if (isnotAnnexB) // 如果不是 AnnexB 格式，需要进行解码
        {
            int ret = av_bsf_send_packet(bsf_ctx, temp_pkt);
            if (ret < 0)
            {
                fprintf(stderr, "Error sending packet to filter\n");
                av_packet_unref(temp_pkt);
                return false;
            }
            ret = av_bsf_receive_packet(bsf_ctx, temp_pkt);
            if (ret < 0)
            {
                fprintf(stderr, "Error receiving packet from filter\n");
                av_packet_unref(temp_pkt);
                return false;
            }
        }

        bool decode_success = decoder.Decode(temp_pkt->data, temp_pkt->size, 0);
        av_packet_unref(temp_pkt);

        if (decode_success)
        {
            status = 0;
            return true;
        }

        no_frame_count++; // 异常处理，已读视频包但解码未出帧的次数
        if (no_frame_count >= MAX_PACKETS_NO_FRAME)
        {
            // 异常：连续多包无输出，避免死循环
            return false;
        }
        std::this_thread::sleep_for(2ms);
    }
}

StreamLoader::StreamLoader(char *url, int id)
{
    stream_loader_id = id;
    std::cout << "StreamLoader: " << std::to_string(id) << std::endl; // 打印流加载器的唯一标识符
    callback = mpp_decoder_frame_callback;                            // 设置解码器的回调函数。
    // mat_ptr = new cv::Mat();

    stream_url = url; // 保存视频文件路径或流地址
    status = 0;       // 初始化状态为成功
    stopFlag = false;
}

StreamLoader::~StreamLoader()
{
    std::cout << "destory stream loader: " << stream_loader_id << std::endl;
    close();
    // delete mat_ptr;
}

int StreamLoader::open()
{
    temp_pkt = av_packet_alloc();
    // av_init_packet is deprecated in FFmpeg 7.x, av_packet_alloc() already initializes the packet
    codecPar = avcodec_parameters_alloc(); // 分配解码器参数结构体
    // pFrame = av_frame_alloc();
    // temp_frame = av_frame_alloc();
    av_dict_set(&options, "rtbufsize", "8192000", 0);   // 设置RTSP缓冲区大小
    av_dict_set(&options, "start_time_realtime", 0, 0); // 设置RTSP流开始时间
    av_dict_set(&options, "rtsp_transport", "tcp", 0);  // 设置RTSP传输协议为TCP
    av_dict_set(&options, "stimeout", "2000000", 0);    // 设置RTSP超时时间
    av_dict_set(&options, "max_delay", "500000", 0);

    // 打开RTSP流
    if (avformat_open_input(&fmtCtx, stream_url, NULL, &options) != 0)
    {
        std::cout << "open rtsp stream failed" << std::endl;
        return -1;
    }
    // 查找RTSP流信息
    if (avformat_find_stream_info(fmtCtx, NULL) < 0)
    {
        return -1;
    }

    // 打印视频相关信息
    av_dump_format(fmtCtx, 0, stream_url, 0);
    // 获取视频的信息
    videoStreamIndex = -1; // 未找到视频流索引，初始化为-1
    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++)
    {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            width = fmtCtx->streams[i]->codecpar->width;
            height = fmtCtx->streams[i]->codecpar->height;
            videoStreamIndex = i;
            break;
        }
    }
    std::cout << "videoindex: " << videoStreamIndex << std::endl;
    if (videoStreamIndex < 0)
    {
        return -2;
    }
    // 根据视频流的编码格式，初始化解码器，并为 H.264 格式配置必要的比特流过滤器。
    AVCodecID rtsp_format = fmtCtx->streams[videoStreamIndex]->codecpar->codec_id;
    if (status == 0)
    {
        int ret = 0;
        void *src_buffer = &(this->buffer);
        switch (rtsp_format)
        {
        case AV_CODEC_ID_H264:
            ret = decoder.Init(264, 25, src_buffer, stream_loader_id);
            // ----------------------------------------------------------
            // 查找H.264比特流过滤器
            bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf)
            {
                fprintf(stderr, "Could not find h264_mp4toannexb filter\n");
                avformat_close_input(&fmtCtx);
                return -3;
            }

            // 初始化比特流过滤器上下文
            if (av_bsf_alloc(bsf, &bsf_ctx) < 0)
            {
                fprintf(stderr, "Could not allocate bsf context\n");
                avformat_close_input(&fmtCtx);
                return -3;
            }
            // 设置过滤器参数
            avcodec_parameters_copy(bsf_ctx->par_in, fmtCtx->streams[0]->codecpar);
            bsf_ctx->time_base_in = fmtCtx->streams[0]->time_base; // 设置输入流的时间基准

            if (av_bsf_init(bsf_ctx) < 0) // 完成过滤器的最终初始化
            {
                fprintf(stderr, "Could not initialize bsf context\n");
                av_bsf_free(&bsf_ctx); // 释放过滤器上下文
                avformat_close_input(&fmtCtx);
                return -3;
            }
            isnotAnnexB = true; // 完成过滤器的最终初始化
            // ----------------------------------------------------------
            std::cout << "H264 " << ret << std::endl; // 打印H.264解码器初始化结果
            break;
        case AV_CODEC_ID_HEVC:
            ret = decoder.Init(265, 25, src_buffer, stream_loader_id);
            std::cout << "HEVC " << ret << std::endl;
            break;
        }
    }

    decoder.SetCallback(this->callback);                                            // 设置解码器的回调函数。
    avcodec_parameters_copy(codecPar, fmtCtx->streams[videoStreamIndex]->codecpar); // 复制视频流的编码参数到解码器参数结构体

    // 获取源视频帧率，用于本地文件限速
    AVStream *st = fmtCtx->streams[videoStreamIndex];
    double fps = av_q2d(st->avg_frame_rate);
    if (fps <= 0)
        fps = av_q2d(st->r_frame_rate);
    if (fps <= 0)
        fps = 25.0;
    source_fps_ = fps;

    is_local_file_ = false;
    if (stream_url)
    {
        std::string u(stream_url);
        if (u.rfind("rtsp://", 0) != 0 && u.rfind("rtmp://", 0) != 0) // 如果不是RTSP或RTMP流
            is_local_file_ = true;
    }

    if (is_local_file_ && source_fps_ > 0) // 如果是本地文件且有源视频帧率
    {
        buffer.throttle = true;                                       // 启用限速
        buffer.frame_interval_ms = (int)(1000.0 / source_fps_ * 1.2); // 计算帧间隔，避免倍速
    }

    return 0;
}
// 循环读取、异常处理、断流检测、智能重连
void StreamLoader::operator()()
{
    while (stopFlag == false) // 如果线程未被停止
    {
        try
        {
            read_frame(); // 读取一帧视频数据
        }
        catch (std::exception &e) // 捕获异常
        {
            std::cout << "exception ............" << std::endl;
            std::cout << e.what() << std::endl;
        }
        if (status) // 如果读取失败
        {
            // status < 0 通常为 AVERROR_EOF（文件播完），触发 reconnect 实现循环播放
            std::cout << "Stream " << stream_loader_id << " EOF, reconnecting..." << std::endl;

            // 先关闭当前流，清理解码器和 AVFormatContext
            close();

            // 根据 URL 类型区分本地文件与网络流：
            // - 本地文件：立即重新 open，相当于从头开始播放，实现循环播放
            // - 网络流（rtsp/rtmp 等）：按原来的逻辑，失败时 10 秒后重试
            bool is_network_stream = false;
            if (stream_url)
            {
                std::string url_str(stream_url); // 转换为字符串
                if (url_str.rfind("rtsp://", 0) == 0 ||
                    url_str.rfind("rtmp://", 0) == 0)
                {
                    is_network_stream = true;
                }
            }

            if (is_network_stream) // 如果是网络流
            {
                // 原有 RTSP 重连逻辑：失败则 10s 后重试
                while (open() != 0) // 如果打开失败
                {
                    std::cout << "Reconnect (network) failed, retry after 10s, id = "
                              << stream_loader_id << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10000)); // 等待10秒后重试
                }
            }
            else
            {
                // 本地文件：立即重新 open，相当于从头开始播放
                // 如果打开失败，短暂等待后快速重试
                while (open() != 0)
                {
                    std::cout << "Reopen local file failed, retry shortly, id = "
                              << stream_loader_id << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                std::cout << "Local file reopened, loop playback, id = "
                          << stream_loader_id << std::endl;
            }

            // 重置状态，继续正常读取帧
            status = 0;
        }
    }
}

// ==================================================================================================

void StreamLoaderManager::load_stream(int id) // 加载视频流
{
    std::cout << "Loading stream id: " << id << std::endl;
    StreamLoader *loader = new StreamLoader(urls[id], id); // 创建视频流加载器，
    loader->open();
    stream_loaders.push_back(loader); //
    threads.emplace_back(std::thread(std::ref(*loader)));
}

// 卸载流
void StreamLoaderManager::unload_stream(int id) // 卸载视频流
{
    std::cout << "Unloading stream id: " << id << std::endl;
    stream_loaders[id]->stopFlag = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (threads[id].joinable()) // 如果线程可加入，确保线程执行存在
        threads[id].join();     // 阻塞等待线程执行完成
    delete stream_loaders[id];  // 删除视频流加载器
}