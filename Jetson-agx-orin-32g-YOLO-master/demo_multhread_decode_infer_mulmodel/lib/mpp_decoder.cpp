/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include "mpp_decoder.h"

MppDecoder::MppDecoder() {}

MppDecoder::~MppDecoder()
{
    if (loop_data.packet) // 1.释放 MPP 数据包（存放压缩数据 H.264/H.265）
    {
        mpp_packet_deinit(&loop_data.packet); // 释放对象，而非内存
        loop_data.packet = NULL;
    }
    if (frame) // 2.释放 MPP 帧（存放解码后的 NV12/YUV 数据）
    {
        mpp_frame_deinit(&frame);
        frame = NULL;
    }
    if (mpp_ctx) // 3.销毁 MPP 解码器上下文（核心资源）
    {
        mpp_destroy(mpp_ctx);
        mpp_ctx = NULL;
    }
    if (loop_data.frm_grp) // 4.释放 MPP 帧组（存放解码后的 NV12/YUV 数据）
    {
        mpp_buffer_group_put(loop_data.frm_grp);
        loop_data.frm_grp = NULL;
    }
}

int MppDecoder::Init(int video_type, int fps, void *userdata, int id) // 初始化硬件解码器
{
    this->id = id;
    MPP_RET ret = MPP_OK;      // MPP返回值遍历，判断是否成功初始化
    this->userdata = userdata; // 保存回调用的数据，事实上是&(this->buffer)
    this->fps = fps;
    this->last_frame_time_ms = 0; // 初始化上一时间
    if (video_type == 264)
    {
        mpp_type = MPP_VIDEO_CodingAVC;
    }
    else if (video_type == 265)
    {
        mpp_type = MPP_VIDEO_CodingHEVC;
    }
    else
    {
        // printf("unsupport video_type %d", video_type);
        return -1;
    }
    // printf("mpi_dec_test start ");
    memset(&loop_data, 0, sizeof(loop_data)); // 清零loop_data结构体
    // printf("mpi_dec_test decoder test start mpp_type %d ", mpp_type);
    MppDecCfg cfg = NULL; //
    MppCtx mpp_ctx = NULL;
    mpp_mpi = NULL;
    ret = mpp_create(&mpp_ctx, &mpp_mpi); // 创建MPP解码器上下文
    if (MPP_OK != ret)
    {
        // printf("mpp_create failed ");
        return 0;
    }
    ret = mpp_init(mpp_ctx, MPP_CTX_DEC, mpp_type); // 创建一个 H.264 或 H.265 解码器，MPP_CTX_DEC 表示解码器上下文，mpp_type 表示编码类型
    if (ret)
    {
        // printf("%p mpp_init failed ", mpp_ctx);
        return -1;
    }
    mpp_dec_cfg_init(&cfg); // 创建解码器配置对象
    /* get default config from decoder context */
    ret = mpp_mpi->control(mpp_ctx, MPP_DEC_GET_CFG, cfg); // MPP_DEC_GET_CFG 获取解码器默认配置
    if (ret)
    {
        // printf("%p failed to get decoder cfg ret %d ", mpp_ctx, ret);
        return -1;
    }
    /*
     * split_parse is to enable mpp internal frame spliter when the input
     * packet is not aplited into frames.
     */
    // 开启 MPP 内部码流切分，让 MPP 能处理输入 packet 不一定刚好一帧的情况。
    ret = mpp_dec_cfg_set_u32(cfg, "base:split_parse", need_split); // 设置解码器配置对象的 split_parse 参数为 1，开启 MPP 内部码流切分功能
    if (ret)
    {
        // printf("%p failed to set split_parse ret %d ", mpp_ctx, ret);
        return -1;
    }
    ret = mpp_mpi->control(mpp_ctx, MPP_DEC_SET_CFG, cfg); // MPP_DEC_SET_CFG，把配置设置给 MPP 解码器
    if (ret)
    {
        // printf("%p failed to set cfg %p ret %d ", mpp_ctx, cfg, ret);
        return -1;
    }
    mpp_dec_cfg_deinit(cfg); // 释放解码器配置对象
    loop_data.ctx = mpp_ctx;
    loop_data.mpi = mpp_mpi;
    loop_data.eos = 0;
    loop_data.frame = NULL;
    return 1;
}

int MppDecoder::Reset() // 重置解码器
{
    if (mpp_mpi != NULL)
    {
        mpp_mpi->reset(mpp_ctx);
    }
    return 0;
}

int MppDecoder::SetCallback(MppDecoderFrameCallback callback)
{
    this->callback = callback; // 保存解码完成后的回调函数
    return 0;
}

int MppDecoder::Decode(uint8_t *pkt_data, int pkt_size, int pkt_eos) // packet如何送入MPP，frame如何取出
{                                                                    // 设置局部变量
    MpiDecLoopData *data = &loop_data;
    RK_U32 pkt_done = 0;
    RK_U32 err_info = 0;
    MPP_RET ret = MPP_OK; // MPP 返回值
    MppCtx ctx = data->ctx;
    MppApi *mpi = data->mpi;
    int got_frames = 0;
    if (packet == NULL)
    {
        ret = mpp_packet_init(&packet, NULL, 0);
    }
    ///////////////////////////////////////////////把FFmpeg AVPacket.data / size包装成MPP packet
    mpp_packet_set_data(packet, pkt_data);   // 设置压缩数据起始大小
    mpp_packet_set_size(packet, pkt_size);   // 设置buffer大小
    mpp_packet_set_pos(packet, pkt_data);    // 设置当前读取位置
    mpp_packet_set_length(packet, pkt_size); // 设置有效数据长度
    // setup eos flag
    if (pkt_eos) // 设置EOS标志位
        mpp_packet_set_eos(packet);
    do
    {
        RK_S32 times = 5;
        // send the packet first if packet is not done
        if (!pkt_done)
        {
            ret = mpi->decode_put_packet(ctx, packet); // 把MPP packet送入MPP解码器
            if (MPP_OK == ret)
                pkt_done = 1;
        }
        // then get all available frame and release
        do
        {
            RK_S32 get_frm = 0;
            RK_U32 frm_eos = 0;
        try_again:
            ret = mpi->decode_get_frame(ctx, &frame); // 尝试从 MPP 解码器取出解码后的图像帧。
            // std::cout << "ret :" << ret << std::endl;
            if (MPP_ERR_TIMEOUT == ret)
            {
                if (times > 0)
                {
                    times--;
                    usleep(2000);
                    goto try_again;
                }
            }
            if (MPP_OK != ret)
            {
                // printf("decode_get_frame failed ret %d ", ret);
                break;
            }
            if (frame) // 如果成功取出解码后的图像帧
            {
                RK_U32 buf_size = mpp_frame_get_buf_size(frame); // 获取解码后的图像帧的内存大小
                if (mpp_frame_get_info_change(frame))            // 当码流分辨率发生变化时，解码器会通过 info_change 标志通知应用程序。
                {
                    // printf("decode_get_frame get info changed found ");
                    if (NULL == data->frm_grp) // 如果图像数据缓冲区组不存在
                    {
                        /* If buffer group is not set create one and limit it */
                        ret = mpp_buffer_group_get_internal(&data->frm_grp, MPP_BUFFER_TYPE_DRM);
                        if (ret) // 如果获取图像数据缓冲区组失败
                        {
                            // printf("%p get mpp buffer group failed ret %d ", ctx, ret);
                            break;
                        }
                        /* Set buffer to mpp decoder */
                        ret = mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, data->frm_grp); // 把图像数据缓冲区组设置给 MPP 解码器
                        if (ret)
                        {
                            // printf("%p set buffer group failed ret %d ", ctx, ret);
                            break;
                        }
                    }
                    else // 如果图像数据缓冲区组存在
                    {
                        /* If old buffer group exist clear it */
                        ret = mpp_buffer_group_clear(data->frm_grp); // 清空图像数据缓冲区组
                        if (ret)                                     // 如果清空图像数据缓冲区组失败
                        {
                            // printf("%p clear buffer group failed ret %d ", ctx, ret);
                            break;
                        }
                    }
                    /* Use limit config to limit buffer count to 24 with buf_size */
                    ret = mpp_buffer_group_limit_config(data->frm_grp, buf_size, 24); // 限制图像数据缓冲区组的内存占用量为24MB
                    if (ret)                                                          // 如果限制失败
                    {
                        // printf("%p limit buffer group failed ret %d ", ctx, ret);
                        break;
                    }
                    /*
                     * All buffer group config done. Set info change ready to let
                     * decoder continue decoding
                     */
                    ret = mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL); // 设置解码器信息改变准备标志位
                    if (ret)                                                      // 如果设置失败
                    {
                        // printf("%p info change ready failed ret %d ", ctx, ret);
                        break;
                    }
                }
                else // 如果解码后的图像帧不是信息改变帧，正常帧，直接提取图像数据
                {
                    // err_info = mpp_frame_get_errinfo(frame) | mpp_frame_get_discard(frame);
                    // if (err_info) {
                    //     // printf("decoder_get_frame get err info:%d discard:%d. ", mpp_frame_get_errinfo(frame), mpp_frame_get_discard(frame));
                    // }
                    //// 1. 获取图像参数
                    RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
                    RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);
                    RK_U32 hor_width = mpp_frame_get_width(frame);
                    RK_U32 ver_height = mpp_frame_get_height(frame);
                    //// 2. 获取时间戳（用于同步）
                    RK_S64 pts = mpp_frame_get_pts(frame);
                    RK_S64 dts = mpp_frame_get_dts(frame);
                    // std::cout<<hor_width<<" "<<ver_height<<" "<<hor_stride<<" "<<ver_stride<<std::endl;
                    // // printf("decoder require buffer w:h [%d:%d] stride [%d:%d] buf_size %d pts=%lld dts=%lld ", hor_width, ver_height, hor_stride,
                    //      ver_stride, buf_size, pts, dts);
                    got_frames++;
                    if (callback != nullptr)
                    {
                        MppFrameFormat format = mpp_frame_get_fmt(frame);                         // 1. 获取帧的像素格式（如 NV12、RGB24 等）
                        char *data_vir = (char *)mpp_buffer_get_ptr(mpp_frame_get_buffer(frame)); // 2. 获取帧的虚拟内存指针，为了后续的图像数据处理
                        // 3. 调用用户设置的回调函数，传递所有图像参数
                        callback(this->userdata, hor_stride, ver_stride, hor_width, ver_height, format, 0, data_vir, this->id);
                    }
                }
                frm_eos = mpp_frame_get_eos(frame); // 4. 获取帧是否为结束帧标志位
                ret = mpp_frame_deinit(&frame);     // 5. 销毁帧对象，释放内部资源
                frame = NULL;
                get_frm = 1; // // 标记本次循环成功获取并处理了一帧
            }
            // if last packet is send but last frame is not found continue
            if (pkt_eos && pkt_done && !frm_eos)
            {
                usleep(1 * 1000);
                continue;
            }
            if (frm_eos)
            {
                // printf("found last frame ");
                break;
            }
            if (get_frm) // 如果本次循环成功获取并处理了一帧
                continue;
            break;
        } while (1);

        if (pkt_done) // 如果数据包缓冲区组已处理完成
            break;
        usleep(3 * 1000);
    } while (1);
    mpp_packet_deinit(&packet);
    return got_frames > 0; // 如果成功获取了一帧图像数据，返回true，否则返回false
}
