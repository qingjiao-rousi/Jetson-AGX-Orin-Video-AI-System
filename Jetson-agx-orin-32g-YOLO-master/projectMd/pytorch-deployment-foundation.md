# PyTorch 学习文档：为大模型部署、端侧部署与 CUDA 学习打基础

## 1. 文档目标

这份文档不是一份“把 PyTorch 全讲完”的百科，而是一份**面向 AI 部署**的前置学习资料。

你的后续目标是：

- 学习 `GPU / CUDA`
- 理解端侧设备上的推理与部署
- 在 `Jetson Orin AGX 32G` 这类板子上跑模型
- 为后续接触 `TensorRT`、量化、模型导出、推理优化打基础

因此，这份文档会有一个明确取舍：

- 会重点讲：`tensor`、`device`、`dtype`、模型结构、推理流程、显存意识、模型保存与导出
- 会适度讲：自动求导、训练循环，因为你需要理解模型是怎么来的
- 不会过度展开：复杂训练技巧、分布式训练、大规模数据并行

一句话概括：

> 你要学的不是“怎么把 PyTorch 当成科研工具用到最花”，而是“怎么把 PyTorch 当成部署前的统一语言学扎实”。

---

## 2. 先建立一张总图

如果把后续的 AI 部署链路看成一条流水线，大致是这样：

1. 用 PyTorch 定义模型
2. 用 PyTorch 加载权重
3. 在 PyTorch 中完成训练或至少完成推理验证
4. 明确输入输出张量的形状、类型、设备位置
5. 把模型转换到更适合部署的格式
6. 再交给 CUDA、TensorRT、NPU 后端或端侧运行时去执行

所以 PyTorch 的角色常常是：

- 模型开发框架
- 推理验证框架
- 导出前的参考基线
- 部署问题排查时的“真值对照”

很多部署问题最后都会回到几个基础问题：

- 输入张量 shape 对不对？
- 数据类型是不是匹配？
- 模型是不是 `eval()` 模式？
- 参数是不是正确加载？
- 算子在目标设备上支不支持？
- 结果偏差是模型本身造成的，还是导出/优化造成的？

这些问题本质上都依赖你对 PyTorch 基础是否扎实。

---

## 3. PyTorch 到底是什么

PyTorch 是一个以 `Tensor` 为核心的数据与计算框架。

你可以把它理解成三层：

1. **数据层**：`torch.Tensor`
2. **计算层**：各种张量运算、神经网络层、损失函数
3. **模型层**：`torch.nn.Module`

它既能做：

- 普通数值计算
- 神经网络训练
- 神经网络推理

也能和底层硬件产生联系：

- CPU
- CUDA GPU
- 某些厂商的 NPU 后端

从部署视角看，PyTorch 最重要的能力不是“训练很方便”，而是：

- 它定义了模型的结构
- 它约定了张量的行为方式
- 它让你能在部署前精确验证输入和输出

---

## 4. Tensor：你后面几乎所有内容的起点

### 4.1 什么是 Tensor

`Tensor` 可以理解成“带有更多属性的多维数组”。

它至少包含以下信息：

- 数据本身
- 维度和形状 `shape`
- 数据类型 `dtype`
- 所在设备 `device`
- 是否参与梯度计算

示例：

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)      # torch.Size([2, 2])
print(x.dtype)      # torch.float32
print(x.device)     # cpu
```

你后面在部署中最常见的问题，很多都和这四个属性直接有关：

- `shape`
- `dtype`
- `device`
- 内存布局

### 4.2 shape 比数值本身更重要

在部署场景里，张量的形状经常比它的具体数值还重要。

例如图像模型里常见输入格式：

```python
[batch, channel, height, width]
```

也就是：

```python
[N, C, H, W]
```

自然语言模型里常见输入可能是：

```python
[batch, seq_len]
```

或者：

```python
[batch, seq_len, hidden_size]
```

如果 shape 理解错了，后面几乎都会错：

- 推理结果错误
- 导出失败
- TensorRT 构图失败
- 动态 batch 配置错误
- 内存占用超预期

所以一定要形成习惯：

> 每次看模型输入输出，先问自己 shape 是什么、每一维代表什么。

### 4.3 dtype 是部署里非常关键的一环

常见数据类型：

- `torch.float32`
- `torch.float16`
- `torch.bfloat16`
- `torch.int8`
- `torch.int64`

在训练里，`float32` 很常见；在部署里，`float16` 和 `int8` 更常见，因为它们更省内存、速度也可能更快。

你需要先建立一个朴素认识：

- `float32`：精度高，内存占用大
- `float16`：精度低一些，但内存占用减半，端侧部署很常见
- `int8`：更省资源，但通常需要量化流程支持

一个简单例子：

```python
x = torch.randn(2, 3, dtype=torch.float32)
y = x.half()

print(x.dtype)  # torch.float32
print(y.dtype)  # torch.float16
```

在 Jetson 这类设备上，`dtype` 的选择会直接影响：

- 显存占用
- 吞吐
- 时延
- 是否能跑得下

### 4.4 device：张量放在哪，决定了它由谁算

PyTorch 中一个张量可以在不同设备上：

- `cpu`
- `cuda:0`
- 某些特定后端设备

例如：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 3).to(device)
```

这里最重要的原则是：

> 参与同一次运算的张量，通常必须在同一个设备上。

例如下面这种写法会报错：

```python
cpu_tensor = torch.randn(2, 3)
gpu_tensor = torch.randn(2, 3, device="cuda")

z = cpu_tensor + gpu_tensor
```

错误本质是：

- 一个在 CPU
- 一个在 GPU
- 不能直接混算

这也是初学部署时最常见的问题之一。

### 4.5 Tensor 的常见创建方式

```python
torch.tensor([1, 2, 3])
torch.zeros(2, 3)
torch.ones(2, 3)
torch.randn(2, 3)
torch.arange(0, 10)
```

这些 API 不难，重点是你要把它们和部署问题联系起来：

- 随机输入可以做推理联调
- 零张量可以做 shape 验证
- 指定 dtype/device 可以模拟真实部署输入

例如：

```python
x = torch.randn(1, 3, 224, 224, dtype=torch.float16, device="cuda")
```

这就已经很接近真实部署输入了。

---

## 5. 张量操作：为什么你必须熟悉

部署并不是“把模型丢进去就结束”，实际工程中会有很多输入前处理、输出后处理。

例如：

- `reshape / view`
- `permute`
- `unsqueeze / squeeze`
- `cat / stack`
- `max / argmax`
- `softmax`

这些操作如果不熟，部署时就容易卡住。

### 5.1 reshape / view

用来调整形状。

```python
x = torch.arange(12)
y = x.reshape(3, 4)
```

使用时你要明确：

- 原始 shape 是什么
- 目标 shape 是什么
- 总元素个数必须一致

### 5.2 unsqueeze / squeeze

经常用于补 batch 维或者去掉多余维度。

```python
x = torch.randn(3, 224, 224)
x = x.unsqueeze(0)   # [1, 3, 224, 224]
```

这在单张图推理中非常常见，因为模型通常要求有 batch 维。

### 5.3 permute

用来交换维度顺序。

```python
x = torch.randn(224, 224, 3)
y = x.permute(2, 0, 1)   # [3, 224, 224]
```

如果你接触图像部署，这个操作要非常熟，因为：

- OpenCV/Numpy 常见图像格式：`HWC`
- PyTorch 卷积模型常见输入格式：`CHW`

### 5.4 cat 和 stack

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

c = torch.cat([a, b], dim=0)    # [4, 3]
d = torch.stack([a, b], dim=0)  # [2, 2, 3]
```

区别是：

- `cat`：沿已有维度拼接
- `stack`：新增一个维度后再拼接

这类概念经常影响模型输入组织方式。

---

## 6. 自动求导：部署不一定天天用，但必须懂

### 6.1 autograd 是什么

PyTorch 很强的地方之一，是它能自动记录计算过程并反向求导。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
y.backward()

print(x.grad)
```

这里：

- `requires_grad=True` 表示要跟踪梯度
- `backward()` 表示反向传播

### 6.2 为什么部署工程师也要懂这个

虽然部署主要做推理，但你仍然必须知道训练态和推理态的区别。

至少要理解：

- 推理时通常不需要梯度
- 不需要梯度时应该关闭 autograd
- 不关闭会浪费显存和计算资源

### 6.3 推理时要用 `torch.no_grad()`

```python
model.eval()
with torch.no_grad():
    output = model(x)
```

这是推理代码的基本写法。

原因是：

- 不记录梯度图
- 更省显存
- 更适合部署前的验证

你后面写任何推理脚本，最好都默认这样写。

---

## 7. Module：模型在 PyTorch 里的组织方式

### 7.1 什么是 `nn.Module`

PyTorch 中一个模型通常继承自 `torch.nn.Module`。

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)
```

你可以把 `Module` 理解成：

- 参数的容器
- 子模块的容器
- 前向计算逻辑的组织方式

### 7.2 参数在哪里

例如：

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

这能帮助你理解：

- 模型有哪些权重
- 每层权重 shape 是什么

对于部署来说，这种理解很有价值，因为导出和权重加载问题常常跟参数结构有关。

### 7.3 `forward()` 很重要

`forward()` 描述了输入如何一步一步变成输出。

从部署视角，它相当于：

- 计算图的语义定义
- 导出前的参考执行逻辑

如果 `forward()` 里有太多 Python 控制流、动态行为、非标准算子，部署和导出通常会更麻烦。

这就是为什么很多“训练时能跑”的模型，到了部署环节会遇到困难。

---

## 8. 训练模式与推理模式：一定要分清

### 8.1 `model.train()` 和 `model.eval()`

PyTorch 模型有两种重要状态：

- `model.train()`
- `model.eval()`

这不是形式上的切换，而是会影响某些层的行为。

最典型的是：

- `Dropout`
- `BatchNorm`

### 8.2 为什么部署前必须 `eval()`

因为部署时我们要的是稳定、确定的推理行为。

```python
model.eval()
with torch.no_grad():
    y = model(x)
```

如果忘记 `eval()`，可能导致：

- 推理结果不稳定
- 和训练后验证结果对不上
- 导出结果与线上结果不一致

这类问题工程里非常常见。

---

## 9. state_dict：理解模型保存和加载的关键

### 9.1 什么是 `state_dict`

`state_dict` 本质上是模型参数的一个字典。

```python
torch.save(model.state_dict(), "model.pth")
```

加载时：

```python
model = SimpleNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

### 9.2 为什么推荐保存 `state_dict`

因为它更清晰、更通用。

你会明确知道：

- 保存的是参数，不是整个 Python 对象
- 模型结构代码和权重是分开的

这更符合工程化和部署化思路。

### 9.3 部署视角下要关注什么

加载模型时要特别检查：

- 权重是否匹配当前模型结构
- 是否存在缺失 key
- 是否存在多余 key
- 权重加载后输出是否正常

如果这里没搞清楚，后面做 CUDA、TensorRT、板端部署时会一直带着隐患。

---

## 10. 一个最小推理流程要长什么样

这是你后面应该形成肌肉记忆的基本模板。

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleNet()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

x = torch.randn(1, 4).to(device)

with torch.no_grad():
    y = model(x)

print(y)
```

这里每一步都有意义：

1. 选择设备
2. 构建模型结构
3. 加载权重
4. 把模型迁移到目标设备
5. 切到推理模式
6. 构造输入
7. 关闭梯度进行推理

后面不管是图像模型、检测模型、分类模型还是 LLM 的某些子模块，基本都离不开这个骨架。

---

## 11. 设备迁移：后面学 CUDA 时最需要打牢的部分

### 11.1 `.to(device)` 是高频操作

```python
model = model.to(device)
x = x.to(device)
```

记住一个硬规则：

> 模型在哪，输入通常也必须在哪。

如果模型在 GPU、输入在 CPU，运行时基本就会报错。

### 11.2 CPU 和 GPU 的差别，不只是速度差

很多初学者会把 GPU 理解成“更快的 CPU”，但在工程上它们还有这些区别：

- 内存空间不同
- 数据传输有成本
- 支持的算子和精度可能不同
- 调试体验不同

所以后面你学 CUDA 时，不要只盯着“加速”，还要关注：

- 数据是否频繁在 CPU/GPU 间拷贝
- 显存是否足够
- kernel 启动开销是否值得
- batch 太小会不会吃不到 GPU 优势

### 11.3 `map_location`

```python
state = torch.load("model.pth", map_location="cpu")
```

这个参数非常实用，因为你可能会遇到：

- 权重在 GPU 环境保存
- 现在却想在 CPU 或 Jetson 上加载

如果不会处理 `map_location`，模型加载经常会出问题。

---

## 12. dtype、精度与端侧部署的关系

### 12.1 为什么精度选择这么重要

端侧设备资源有限，尤其你在板子上做部署时，往往会关注：

- 显存够不够
- 推理速度够不够
- 功耗是否可接受

这时 `float16` 往往比 `float32` 更有部署价值。

### 12.2 一个直观理解

如果某个张量有同样数量的元素：

- `float32` 一般每个元素 4 字节
- `float16` 一般每个元素 2 字节

也就是说，在理想情况下，`float16` 内存占用大约减半。

这对大模型和端侧部署非常关键。

### 12.3 但不是所有情况都能无脑降精度

你要有这个意识：

- 精度降低可能带来数值误差
- 某些算子可能对低精度更敏感
- 训练与推理的精度策略可能不同

所以部署里常见流程不是“盲目转半精度”，而是：

1. 先有 `float32` 基线结果
2. 再尝试 `float16`
3. 对比速度、显存和输出误差

这就是工程思维。

---

## 13. batch 的概念：吞吐与时延的基础

### 13.1 什么是 batch

`batch` 就是一次送入模型的样本数。

例如图像分类：

- `batch=1`：一次只推理一张图
- `batch=8`：一次推理 8 张图

### 13.2 为什么部署时特别关注 batch

因为它直接影响：

- 吞吐量
- 单次时延
- 显存占用

一般来说：

- 大 batch 可能提高吞吐
- 小 batch 更接近实时系统
- batch 太大会爆显存

在 Jetson 场景下，很多任务更偏：

- `batch=1`
- 低时延
- 稳定功耗

所以你后面做端侧部署，不能只盯着“每秒处理多少”，还要看：

- 单帧时延
- 峰值内存
- 板子是否持续稳定运行

---

## 14. 显存与内存意识：从现在就要养成

### 14.1 为什么这点重要

大模型部署、端侧部署，很多时候最先撞上的不是精度问题，而是资源问题。

你需要区分：

- 主机内存 RAM
- GPU 显存 VRAM
- Jetson 上的共享内存体系和可用资源约束

虽然 PyTorch 帮你管理了很多细节，但你必须形成“资源敏感”意识。

### 14.2 什么东西会吃内存/显存

- 模型参数
- 中间激活
- 输入输出张量
- 梯度
- 优化器状态

部署时通常不需要梯度和优化器，因此推理一般比训练更省资源。

这也是为什么部署脚本一定要：

- `model.eval()`
- `torch.no_grad()`

### 14.3 对 Jetson 的现实认识

Jetson Orin AGX 32G 很强，但它不是数据中心 GPU。

你后续做端侧部署时，需要更敏感地处理：

- 模型大小
- 输入分辨率
- batch 大小
- 精度格式
- 后处理是否过重

如果前面这些 PyTorch 基础没打好，到了板子上问题会很杂，而且不好定位。

---

## 15. 为什么你要先学 PyTorch，再学 CUDA / TensorRT / NPU

这是一个非常关键的认知点。

很多人会想直接冲 CUDA 或 TensorRT，但如果 PyTorch 基础不牢，会出现这些问题：

- 不知道模型真正输入输出是什么
- 看到 shape 错误不会定位
- 结果不一致时无法判断是模型问题还是部署问题
- 只会“照着教程点按钮”，无法独立排查

PyTorch 是你的“参考实现层”。

后面不管你把模型部署到：

- CUDA
- TensorRT
- ONNX Runtime
- 某种 NPU SDK

你都需要先在 PyTorch 里回答这几个问题：

1. 模型结构是什么？
2. 输入 shape、输出 shape 是什么？
3. 输入 dtype、输出 dtype 是什么？
4. 基线结果是什么？
5. 误差容忍范围是什么？

---

## 16. 面向大模型部署时，PyTorch 哪些点最重要

如果你的方向偏大模型部署，而不是传统小 CNN，那么下面几个点更重要。

### 16.1 张量维度理解能力

大模型中很常见的维度概念：

- `batch`
- `seq_len`
- `hidden_size`
- `num_heads`
- `head_dim`

如果维度感不强，后面看 attention、KV cache、prefill、decode 都会比较吃力。

### 16.2 dtype 和显存估算

大模型部署非常依赖：

- `fp32`
- `fp16`
- `bf16`
- `int8`
- `int4`

哪怕你现在还没进入量化细节，也应该先建立意识：

- 精度不仅影响结果，也影响资源占用
- 模型越大，dtype 选择越关键

### 16.3 推理态思维

大模型部署更关注：

- `eval()`
- `no_grad()`
- token-by-token 推理
- cache 复用
- 延迟和吞吐权衡

即使你现在先学的是基础 PyTorch，这套推理思维也要提前建立。

---

## 17. 面向 Jetson Orin AGX 32G，你现在应重点具备的 PyTorch 能力

结合你的设备，建议你先把以下能力练扎实：

### 17.1 会写标准推理脚本

你应该能独立写出：

- 加载模型
- 加载权重
- 切换 `eval()`
- 构造输入
- 放到 `cuda`
- 执行推理
- 打印输出 shape 和耗时

### 17.2 会检查输入输出

至少要养成打印这些信息的习惯：

```python
print(x.shape, x.dtype, x.device)
print(y.shape, y.dtype, y.device)
```

很多板端问题，本质上就是这里没有检查清楚。

### 17.3 会建立基线

例如：

- CPU 上结果是什么
- GPU 上结果是什么
- `fp32` 和 `fp16` 差多少

后面当你切 TensorRT、量化或 NPU 后端时，这些基线会非常有用。

### 17.4 会做最基本的性能测试

比如：

- 单次推理耗时
- 多次 warmup 后平均耗时
- batch=1 和 batch=4 的差异

这能帮你逐渐形成部署思维，而不是停留在“能跑就行”。

---

## 18. 一个更贴近部署的 PyTorch 推理模板

```python
import time
import torch
import torch.nn as nn


class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = DemoNet().to(device)
    model.eval()

    x = torch.randn(1, 1024, device=device, dtype=dtype)
    if dtype == torch.float16:
        model = model.half()

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

        start = time.time()
        y = model(x)
        end = time.time()

    print("input :", x.shape, x.dtype, x.device)
    print("output:", y.shape, y.dtype, y.device)
    print("latency:", (end - start) * 1000, "ms")


if __name__ == "__main__":
    main()
```

这个模板体现了几个很重要的部署意识：

- 明确设备
- 明确 dtype
- 使用推理模式
- 做 warmup
- 打印输入输出属性
- 测试时延

---

## 19. 学习中最容易犯的错误

### 19.1 只看代码，不看张量属性

很多人会盯着模型代码看半天，却不打印：

- `shape`
- `dtype`
- `device`

这会让调试效率非常低。

### 19.2 忘记 `eval()` 或 `no_grad()`

这会导致：

- 性能不对
- 显存浪费
- 推理结果不稳定

### 19.3 不理解数据预处理

哪怕模型本体是对的，如果输入预处理错了，结果也会错。

例如：

- 通道顺序错了
- 归一化错了
- batch 维忘了加

### 19.4 没有建立 PyTorch 基线

后面做导出或部署优化时，如果没有基线，你很难知道问题出在哪一层。

---

## 20. 推荐的学习顺序

如果你的目标是“大模型部署 + 端侧部署 + Jetson + CUDA”，建议按这个顺序学：

### 第一阶段：PyTorch 核心基础

重点掌握：

- Tensor
- shape
- dtype
- device
- Module
- forward
- eval/train
- no_grad
- state_dict

目标：

> 能独立写出一个标准的 PyTorch 推理脚本。

### 第二阶段：部署前置能力

重点掌握：

- 输入输出检查
- 性能测试
- `fp32/fp16` 对比
- batch 对时延和显存的影响
- 模型保存与加载

目标：

> 能在本地把模型推理流程跑顺，并知道如何验证结果。

### 第三阶段：CUDA 与 GPU 基础

重点掌握：

- GPU 和 CPU 内存差异
- Host 到 Device 的数据搬运
- CUDA 执行模型基础
- 为什么并行能快，为什么不一定总快

目标：

> 能理解 PyTorch 的 `.to("cuda")` 背后大概发生了什么。

### 第四阶段：Jetson 端侧部署

重点掌握：

- Jetson 环境配置
- CUDA/cuDNN/TensorRT 基本关系
- 板端性能观察
- batch、功耗、温度、稳定性

目标：

> 能在 Jetson 上独立完成一个模型从 PyTorch 验证到板端推理的闭环。

### 第五阶段：模型导出与优化

重点掌握：

- ONNX
- TensorRT
- 半精度
- 量化
- 算子兼容性

目标：

> 能把“模型能跑”进一步推进到“模型跑得快、占用低、可稳定部署”。

---

## 21. 你接下来非常值得动手练的练习

下面这些练习非常适合作为 PyTorch 到部署的过渡。

### 练习 1：张量属性练习

要求自己熟练写出并解释：

- 创建不同 shape 的张量
- 切换不同 dtype
- 在 CPU/GPU 上迁移
- 打印 shape、dtype、device

目标：

> 看到一个张量时，你能第一时间说清楚它长什么样、在哪里、用什么精度。

### 练习 2：最小 MLP 推理脚本

写一个两层全连接网络，完成：

- 定义模型
- 构造输入
- 放到 GPU
- `eval()`
- `no_grad()`
- 测试耗时

目标：

> 形成标准推理模板。

### 练习 3：`fp32` 与 `fp16` 对比

比较：

- 输出差异
- 显存差异
- 速度差异

目标：

> 建立部署中的精度与性能权衡意识。

### 练习 4：图像输入维度转换

从 `HWC` 转成 `NCHW`，并补 batch 维。

目标：

> 为后续视觉模型部署打基础。

### 练习 5：保存与加载权重

完成：

- `state_dict` 保存
- 新建同结构模型重新加载
- 验证推理输出

目标：

> 理解模型结构和权重文件是两回事。

---

## 22. 这一阶段你应该达到什么水平

学完这份文档后，你不一定已经会部署大模型，但你应该达到下面这个状态：

- 知道 PyTorch 的核心对象是什么
- 能理解模型输入输出张量
- 能写基本推理代码
- 知道 `eval()` 和 `no_grad()` 的意义
- 知道 `device` 和 `dtype` 在部署中为什么重要
- 能从“训练框架视角”切换到“部署框架视角”

如果这些都能做到，后面继续学：

- CUDA
- Jetson
- TensorRT
- 量化
- ONNX

就会顺很多。

---

## 23. 一份非常实用的自查清单

每次你写一个 PyTorch 推理脚本，都可以拿这份清单过一遍：

- 模型是否已经 `model.eval()`？
- 推理是否放在 `torch.no_grad()` 中？
- 输入 shape 是否正确？
- 输入 dtype 是否正确？
- 输入和模型是否在同一个 device？
- 权重是否成功加载？
- 输出 shape 是否符合预期？
- 是否建立了 `fp32` 基线？
- 是否测试了时延？
- 是否考虑了板端内存/显存约束？

如果这些检查习惯养成，后面做端侧部署会省很多时间。

---

## 24. 总结

对于你接下来的学习路线来说，PyTorch 的价值不只是“会写模型”，而是它提供了一套共同语言：

- 用 `Tensor` 描述数据
- 用 `Module` 描述模型
- 用 `device` 描述算在哪里
- 用 `dtype` 描述精度与资源
- 用标准推理流程验证部署前的正确性

你后面学到的很多高级内容，本质上都会回到这些问题：

- 数据长什么样？
- 数据在哪？
- 用什么精度？
- 模型怎么执行？
- 执行代价是多少？

如果这几个问题你能在 PyTorch 里讲清楚，那么你就已经为后续的：

- CUDA 学习
- Jetson 部署
- 大模型推理优化
- TensorRT 和端侧工程实践

打下了非常关键的前置基础。

---

## 25. 建议的下一步

建议你接下来按这个顺序继续学：

1. 用 PyTorch 手写 3 到 5 个最小推理脚本
2. 练熟 `shape / dtype / device / eval / no_grad`
3. 在有 GPU 的环境下练习 `.to("cuda")`
4. 对比 `fp32` 和 `fp16`
5. 再进入 CUDA、Jetson、TensorRT

如果你愿意，下一步我可以继续帮你生成两份配套资料：

1. `PyTorch 最小可运行示例集.md`
2. `Jetson Orin AGX 32G 部署学习路线图.md`

这两份会和当前文档直接衔接，适合你继续往下学。
