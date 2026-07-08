/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#ifndef _rknnPool_H
#define _rknnPool_H

#include <queue>
#include <vector>
#include <iostream> // 用于调试输出
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rknn_api.h" //RKNN Runtime API, 用于加载和推理RKNN模型
#include "postprocess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "ThreadPool.hpp" //线程池, 用于并行处理视频帧
using cv::Mat;
using std::queue;
using std::vector;

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);       // 读取文件数据
static unsigned char *load_model(const char *filename, int *model_size); // 读取模型文件数据，返回大小＋数据指针

class rknn_lite
{
private:
    rknn_context rkModel;           // RKNN模型上下文, 用于加载和推理模型
    unsigned char *model_data;      //.rknn模型文件读取到的内存地址
    rknn_sdk_version version;       // RKNN SDK版本信息
    rknn_input_output_num io_num;   // 输入输出参数数量
    rknn_tensor_attr *input_attrs;  // 输入tensor属性
    rknn_tensor_attr *output_attrs; // 输出tensor
    rknn_input inputs[1];           // 推理输入参数, 用于存储输入图像的tensor
    int ret;                        // 推理返回值
    int channel = 3;
    int width = 0;
    int height = 0;
    int class_num = 0; // 类别数量,表示
    int id;

public:
    Mat ori_img;                                            // 当前要推理的图像
    int interf(detect_result_group_t &detect_result_group); // 推理函数, 输入当前图像, 输出检测结果组
    rknn_lite(char *dst, int n, int class_num, int id);     // 构造函数, 初始化rknn类
    ~rknn_lite();
};

rknn_lite::rknn_lite(char *model_name, int n, int class_num, int id) // 构造函数, 初始化rknn类，n为核心号
{
    this->class_num = class_num;
    this->id = id;
    /* Create the neural network */
    printf("Loading model id = %d\n", id);
    int model_data_size = 0; // 初始化为0, 用于存储模型文件的大小
    // 读取模型文件数据
    model_data = load_model(model_name, &model_data_size);
    // 通过模型文件初始化rknn类
    ret = rknn_init(&rkModel, model_data, model_data_size, 0, NULL); // 创建rknn模型上下文
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    rknn_core_mask core_mask; // 枚举类型, 用于指定核心号
    if (n == 0)
        core_mask = RKNN_NPU_CORE_0; // 指定核心号为0
    else if (n == 1)
        core_mask = RKNN_NPU_CORE_1;
    else
        core_mask = RKNN_NPU_CORE_2;
    ret = rknn_set_core_mask(rkModel, core_mask); // 设置核心号
    if (ret < 0)
    {
        printf("rknn_set_core_mask error ret=%d\n", ret);
        exit(-1);
    }

    // 初始化rknn类的版本
    ret = rknn_query(rkModel, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version)); // 获取当前RKNN API的软件版本号和 NPU 驱动版本号
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret); // 获取版本号失败, 退出程序
        exit(-1);
    }

    // 获取模型的输入参数
    ret = rknn_query(rkModel, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // 设置输入数组
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr)); // 获取输入参数的属性
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
    }

    // 设置输出数组
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs)); // 初始化输出参数的属性为0
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr)); // 获取输出参数的属性
    }
    // 描述模型输入和输出张量（Tensor）的元数据
    //  设置输入参数
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) // 如果输入参数的格式为NCHW
    {
        channel = input_attrs[0].dims[1]; // 从第2个维度获取通道数
        height = input_attrs[0].dims[2];  // 从第3个维度获取高度
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    // 准备RKNN推理的输入参数, 用于存储输入图像的tensor
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
}

rknn_lite::~rknn_lite()
{
    ret = rknn_destroy(rkModel);
    delete[] input_attrs;
    delete[] output_attrs;
    if (model_data)
        free(model_data);
}
// 推理函数, 输入当前图像, 输出检测结果组
int rknn_lite::interf(detect_result_group_t &detect_result_group)
{
    cv::Mat img; // 输入图像的tensor
    // 获取图像宽高
    int img_width = ori_img.cols;
    int img_height = ori_img.rows;
    cv::cvtColor(ori_img, img, cv::COLOR_BGR2RGB); // 将图像转换为RGB格式，模型通常用RGB训练
    if (img_width != width || img_height != height)
        cv::resize(img, img, cv::Size(width, height)); // 缩放到模型输入尺寸，模型输入尺寸固定（如640x640）
    inputs[0].buf = (void *)img.data;                  // 将图像数据指针赋值给输入缓冲区
    // 设置rknn的输入数据
    rknn_inputs_set(rkModel, io_num.n_input, inputs); //

    // 设置输出
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
        outputs[i].want_float = 0; // 调用npu进行推演
    ret = rknn_run(rkModel, NULL);
    // 获取npu的推演输出结果
    ret = rknn_outputs_get(rkModel, io_num.n_output, outputs, NULL);
    // 后处理输出结果和可视化结果
    const float nms_threshold = NMS_THRESH;      // 非极大值抑制阈值
    const float box_conf_threshold = BOX_THRESH; // 箮信度阈值
    // 把NPU输出的INT8整数，转换回有意义的浮点数
    float scale_w = (float)width / img_width;
    float scale_h = (float)height / img_height; // 缩放比例

    std::vector<float> out_scales;            // 存放每个输出张量的缩放因子（scale）
    std::vector<int32_t> out_zps;             // 存放每个输出张量的偏移量（zp）
    for (int i = 0; i < io_num.n_output; ++i) // 数量：等于模型输出个数
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    // 调用YOLO后处理函数，把 RKNN 输出的 3 个尺度 tensor 转成最终检测框。
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group, class_num);

    // 绘制检测框和文本标签
    char text[256];
    // std::cout << detect_result_group.count << " objects detected.\n";
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        if (id == 0)
            rectangle(ori_img, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 255, 0, 0), 3);
        else
        {
            rectangle(ori_img, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 0, 255, 0), 3);
        }
        // putText(ori_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }

    ret = rknn_outputs_release(rkModel, io_num.n_output, outputs);

    return 0;
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) // 从文件指定位置读取指定大小数据到内存
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET); // 将文件指针移动到指定位置
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size) // 读取模型文件数据，返回大小＋数据指针
{
    FILE *fp;            // 文件指针, 用于读取模型文件数据
    unsigned char *data; // 模型文件数据指针, 用于存储读取到的模型文件数据

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END); //// 将文件指针移动到文件末尾
    int size = ftell(fp);

    data = load_data(fp, 0, size); // 用于读取模型文件数据

    fclose(fp);

    *model_size = size;
    return data;
}

#endif