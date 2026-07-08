/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#pragma once
extern "C"
{
    // 以c语言的方式包含avformat库的头文件，用于读取视频流，如下都是ffmpeg的头文件
#include <libavformat/avformat.h> //打开视频流、读取封装流、获取stream信息
#include <libavcodec/avcodec.h>   //解码视频流
#include <libavcodec/bsf.h>       //比特流过滤器，用于处理H264流的Annex B格式问题
#include <libavutil/imgutils.h>   //图像工具函数，用于处理图像数据
#include <libavutil/rational.h>   //理数工具函数，用于处理时间戳
#include <libavutil/time.h>       //时间工具函数，用于处理时间戳
#include <libswscale/swscale.h>   //图像缩放函数，用于缩放图像
}
#include "mpp_decoder.h"      //mpp硬解码器封装
#include <opencv2/opencv.hpp> //cv::Mat，用于存储图像数据
#include <thread>             //线程
#include <atomic>             //原子操作
#include <functional>         // std::function，用于定义回调函数类型
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
using std::queue;
using std::vector;
#include "m_buffer.hpp" //图像数据缓冲区

using MppDecoderFrameCallback = std::function<void(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data, int id)>;

class StreamLoader
{
public:
    MppDecoder decoder; // 当前这一路视频对应的 MPP 硬解码器。
    // 视频流的数据起始地址的索引
    // ffmpeg相关的输入成员
    int videoStreamIndex;               // 视频流的索引
    AVDictionary *options = NULL;       // 保存RTSP地址，例如tcp/超时时间等
    AVFormatContext *fmtCtx = NULL;     // 输入上下文，用于读取视频流
    AVCodecParameters *codecPar = NULL; // 编码参数，

    AVBSFContext *bsf_ctx = NULL; // 比特流过滤器上下文，用于处理H264流的Annex B格式问题
    const AVBitStreamFilter *bsf;

    // 是否已经读取到关键帧
    bool got_key_frame = false;
    // 存储接受流数据的临时结构，内存在构造函数中申请
    AVPacket *temp_pkt; // 读取到的包
    // 当前包的编号
    int current_pkt_id = 0;
    // 当前对象的唯一标识号，第几路视频
    int stream_loader_id;
    // 流地址
    char *stream_url = nullptr; // 视频流地址
    int width = 0;
    int height = 0;
    int status = 0;
    bool isnotAnnexB = false;         // H.264 bitstream filter 相关成员
    MppDecoderFrameCallback callback; // 解码器回调函数，用于处理解码后的图像数据

    // 管理图像数据
    Mbuffer buffer; // 图像数据缓冲区

    std::atomic<bool> stopFlag; // 停止标志位，用于停止线程

    // 本地文件播放时按源视频帧率限速，避免倍速
    double source_fps_ = 25.0;
    bool is_local_file_ = false; // 是否是本地文件

    void close();
    bool read_frame();
    StreamLoader(char *url, int id); // 构造函数
    ~StreamLoader();                 // 析构
    int open();
    void operator()();   // 线程入口函数，循环读取视频流
    void update_queue(); // 更新图像数据缓冲区
};

class StreamLoaderManager // 一个单例类，只有一个流管理器实例，用于管理多个视频流的加载和卸载
{
public:
    // -----------------------------------------------
    // 本地测试用例
    // char *url105 = "rtsp://admin:jhx12345@192.168.1.105:554/Streaming/Channels/101";
    // char *url104 = "rtsp://admin:jhx12345@192.168.1.104:554/Streaming/Channels/101";
    // vector<char *> urls = {url105, url104, url105, url104, url105, url104};
    char *url1 = "../../1.mp4";
    char *url2 = "../../2.mp4";
    char *url3 = "../../3.mp4";
    char *url4 = "../../4.mp4";
    vector<char *> urls = {url1, url2, url3, url4, url2, url3};
    int num_stream = 4;
    // -----------------------------------------------
    // 禁止拷贝构造和赋值操作
    StreamLoaderManager(const StreamLoaderManager &) = delete;
    StreamLoaderManager &operator=(const StreamLoaderManager &) = delete;

    // 获取单例实例的静态方法
    static StreamLoaderManager &getInstance()
    {
        static StreamLoaderManager instance; // C++11 保证了静态局部变量的线程安全性
        return instance;
    }

    // 加载流
    void load_stream(int id);
    // 停止流
    // 该函数没有使用
    void unload_stream(int id);

    vector<StreamLoader *> stream_loaders;
    vector<std::thread> threads; // 储存每个视频流的线程对象

private:
    // 私有构造函数，防止从外部创建对象
    StreamLoaderManager()
    {
        std::cout << "StreamLoaderManager created" << std::endl;
    }

    // 私有析构函数，防止外部删除对象
    ~StreamLoaderManager()
    {
        std::cout << "StreamLoaderManager destroyed" << std::endl;
    }
};