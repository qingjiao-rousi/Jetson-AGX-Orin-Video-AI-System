
/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include "stream_loader.h"            //负责拉流读视频
#include "rknnPool.hpp"               //负责RKNN/NPU的推理封装
#include "streaming_manager.h"        //负责推流管理
#include "detection_fusion_manager.h" //负责融合多个检测模型的结果
#include "im2d.h"                     //这是RGA接口，用来做硬件缩放，

char *model_person = "../../model/person_relu.rknn";
char *model_helmet = "../../model/helmet_relu.rknn";
// char *model_tired = "../../model/tired_relu.rknn";
char *model_callplay = "../../model/callplay_relu.rknn";           // 可以使用const char*,因为这些字符串本身不会被修改
StreamLoaderManager &manager = StreamLoaderManager::getInstance(); // 单例管理器，只有这一个实例，全局共享
// 创建RKNN模型的集合，用于存储多个模型实例
vector<rknn_lite *> rk_pool;
// 创建线程池对象，使用n个线程
vector<std::thread> rk_threads;
// 用于存储显示图像的Mat
vector<cv::Mat> images(6);
// 管理images的互斥锁
vector<std::mutex> mutexes(6);

// 推流管理器
StreamingManager streaming_manager; // 负责推流管理

// 检测融合管理器
DetectionFusionManager fusion_manager; // 把person、helmet、callplay的检测结果融合起来

void combineImage(StreamLoaderManager &manager) // 合并所有流的图像拼成一张大图交给推流模块
{
    cv::Mat combinedImage(1080, 1280, CV_8UC3, cv::Scalar(0, 0, 0)); // 初始化为黑色
    cv::Mat lastCombinedImage;                                       // 保存最后一帧
    bool hasLastFrame = false;
    const int target_fps = 24;                               // 目标帧率
    const int frame_interval_ms = 1000 / target_fps;         // 每帧间隔（毫秒）
    auto last_frame_time = std::chrono::steady_clock::now(); // 上一帧的时间

    while (true)
    {
        auto current_time = std::chrono::steady_clock::now(); // 当前时间
        // 计算上一帧到当前帧的时间间隔（毫秒）
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_frame_time).count();

        // 控制帧率：如果还没到时间，等待
        if (elapsed < frame_interval_ms)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_interval_ms - elapsed));
            current_time = std::chrono::steady_clock::now();
        }
        last_frame_time = current_time; // 更新上一帧时间

        bool hasNewFrame = false;
        const int tile_w = 640, tile_h = 360; // 每个流的图像大小

        for (int i = 0; i < manager.num_stream; ++i) // 遍历所有视频流
        {
            cv::Mat local_img;
            {
                std::lock_guard<std::mutex> lock(mutexes[i]); // 加锁，确保线程安全
                // 检查是否有新的图像可处理
                if (images[i].empty())
                    continue;
                local_img = std::move(images[i]); // 移动语义，避免拷贝构造
                images[i] = cv::Mat();
            }

            cv::Mat resizedImage(tile_h, tile_w, CV_8UC3); // 初始化为黑色
            int src_w = local_img.cols, src_h = local_img.rows;
            // 用RGA进行缩放
            rga_buffer_t src_buf = wrapbuffer_virtualaddr(local_img.data, src_w, src_h, RK_FORMAT_BGR_888);      // 源图包装
            rga_buffer_t dst_buf = wrapbuffer_virtualaddr(resizedImage.data, tile_w, tile_h, RK_FORMAT_BGR_888); // 目标图包装
            // 把opencv图像包装成RGA能识别的缓冲区，然后调用RGA把原图缩放到目标图大小
            IM_STATUS status = imresize(src_buf, dst_buf);
            if (status != IM_STATUS_SUCCESS)
            {
                cv::resize(local_img, resizedImage, cv::Size(tile_w, tile_h)); // 如果失败就退回opencv的resize缩放
            }

            int row = i / 2, col = i % 2;
            int x = col * tile_w, y = row * tile_h;
            resizedImage.copyTo(combinedImage(cv::Rect(x, y, tile_w, tile_h)));
            hasNewFrame = true;
        }

        cv::Mat frameToSend; // 用于发送的帧
        if (hasNewFrame)
        {
            lastCombinedImage = combinedImage.clone(); // 复制当前帧
            frameToSend = lastCombinedImage;           // 发送当前帧
            hasLastFrame = true;
        }
        else if (hasLastFrame) // 如果有上一帧
        {
            frameToSend = lastCombinedImage; // 发送上一帧
        }
        else
        {
            frameToSend = combinedImage.clone(); // 复制当前帧
        }

        // 显示合成图像
        // cv::imshow("Combined Image", frameToSend);

        // 发送到推流管理器
        StreamingData stream_data;                                // 推流数据结构体
        stream_data.stream_id = 0;                                // 流ID，这里设为0
        stream_data.frame = frameToSend;                          // 帧数据
        stream_data.timestamp = std::chrono::system_clock::now(); // 当前时间戳
        // 初始化检测结果（如果需要绘制检测结果，需要从推理结果中获取）
        memset(&stream_data.person_results, 0, sizeof(detect_result_group_t));
        memset(&stream_data.helmet_results, 0, sizeof(detect_result_group_t));
        memset(&stream_data.tired_results, 0, sizeof(detect_result_group_t));
        memset(&stream_data.callplay_results, 0, sizeof(detect_result_group_t));

        streaming_manager.addStreamingData(stream_data);

        // 等待按键
        // if (cv::waitKey(10) == 27)
        // break;
    }
    cv::destroyAllWindows(); // 销毁所有窗口
}
// 多路视频ai推理线程
void rknn_infer(rknn_lite *p1, rknn_lite *p2, rknn_lite *p3, rknn_lite *p4, int i) // 每一路的推理线程
{
    dpool::ThreadPool pool(4); // 4个线程,用线程池是因为一张图要跑多个模型：person、helmet、tired、callplay
                               // 线程里再并行模型任务，每个模型一个线程，提高推理效率
    detect_result_group_t g1, g2, g3, g4;
    memset(&g1, 0, sizeof(detect_result_group_t));
    memset(&g2, 0, sizeof(detect_result_group_t));
    memset(&g3, 0, sizeof(detect_result_group_t));
    memset(&g4, 0, sizeof(detect_result_group_t));

    // 跳帧推理：每 INFER_INTERVAL 帧做一次完整推理，中间帧复用上一帧检测结果，提高实时性
    const int INFER_INTERVAL = 2; // 每2帧做一次完整推理
    int frame_count = 0;          // 当前帧计数
    std::vector<FusedDetection> last_fused;

    while (!manager.stream_loaders[i]->stopFlag)
    {
        std::unique_lock<std::mutex> lock(manager.stream_loaders[i]->buffer.mtx); // 没收到信号就无限处理视频帧
        if (manager.stream_loaders[i]->buffer.img.empty())                        // 如果视频帧为空
        {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        p1->ori_img = manager.stream_loaders[i]->buffer.img.clone();
        p2->ori_img = p1->ori_img;
        if (p3)
            p3->ori_img = p1->ori_img;
        p4->ori_img = p1->ori_img;
        lock.unlock();

        frame_count++;
        bool do_infer = (frame_count % INFER_INTERVAL == 1) || last_fused.empty();

        if (do_infer) // 如果需要做推理
        {
            auto f1 = pool.submit([&]()
                                  { p1->interf(g1); }); // 提交person模型推理任务
            auto f2 = pool.submit([&]()
                                  { p2->interf(g2); });
            std::future<void> f3;
            if (p3)
                f3 = pool.submit([&]()
                                 { p3->interf(g3); });
            auto f4 = pool.submit([&]()
                                  { p4->interf(g4); });

            f1.get();
            f2.get();
            if (p3)
                f3.get();
            f4.get(); // 等待callplay模型推理任务完成

            last_fused = fusion_manager.fuseDetections(g1, g2, g3, g4); // 合并所有模型的检测结果
        }

        fusion_manager.drawFusedDetections(p1->ori_img, last_fused); // 绘制合并后的检测结果

        std::unique_lock<std::mutex> lockimage(mutexes[i]); // 加锁，确保线程安全
        images[i] = std::move(p1->ori_img);                 // 复制当前帧到共享内存
        lockimage.unlock();                                 // 解锁，允许其他线程访问共享内存
    }
}

int main(int argc, char *argv[]) // 程序启动主函数
{
    manager.num_stream = std::stoi(argv[1]); // 获取输入视频流数量

    // 初始化推流配置，这是推流模块的结构体，用于配置推流参数，如RTMP地址、视频尺寸、帧率、码率等
    StreamingConfig stream_config;
    stream_config.rtmp_url = "rtmp://192.168.137.1/live/livestream"; // 替换为实际的RTMP地址
    stream_config.width = 1280;
    stream_config.height = 720;
    stream_config.fps = 24;
    stream_config.bitrate = 2000000;
    stream_config.enable_rtmp = true;
    stream_config.draw_detections = true;

    // 初始化推流管理器
    if (!streaming_manager.initialize(stream_config)) // 如果初始化失败
    {
        std::cerr << "Failed to initialize streaming manager" << std::endl;
        return -1;
    }

    // 启动推流
    streaming_manager.startStreaming(); // 启动推流

    // 解码与推理线程
    for (int i = 0; i < manager.num_stream; ++i) // 遍历每个视频流
    {
        manager.load_stream(i);
        // 不同模型绑定不同 NPU 核心，使 ThreadPool 内 person/helmet/callplay 可并行推理
        rknn_lite *ptr1 = new rknn_lite(model_person, 0, 1, 0);
        rknn_lite *ptr2 = new rknn_lite(model_helmet, 1, 2, 1);
        rknn_lite *ptr3 = nullptr;
        rknn_lite *ptr4 = new rknn_lite(model_callplay, 2, 2, 3);
        rk_threads.push_back(std::thread(rknn_infer, ptr1, ptr2, ptr3, ptr4, i)); // 为这个视频流创建一个推理线程
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 等待500ms，确保所有视频流都加载完成
    std::thread readerThread(combineImage, std::ref(manager));   // 创建一个线程，用于合并视频帧
    readerThread.join();                                         // 等待合并线程完成

    // 停止推流
    streaming_manager.stopStreaming(); // 停止推流

    for (int i = 0; i < manager.num_stream; ++i)
    {
        manager.unload_stream(i);
    }
    for (auto &t : rk_threads)
    {
        if (t.joinable())
            t.join();
    }

    return 0;
}
