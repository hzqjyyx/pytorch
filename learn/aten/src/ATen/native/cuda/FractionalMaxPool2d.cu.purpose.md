这个文件实现了分数最大池化（Fractional Max Pooling）的CUDA前向计算。

**核心概念：**
- 分数最大池化是一种随机池化方式，与标准最大池化不同，它在池化窗口大小和位置上引入随机性
- 使用随机采样值来确定每个输出点对应的池化窗口在输入中的位置

**主要组件：**

- **`get_interval()` 函数**
  - 根据随机样本、输出索引等参数计算池化窗口在输入中的起始位置
  - 使用公式：`alpha = (inputSize - poolSize) / (outputSize - 1)` 将输出坐标映射到输入坐标

- **`fractional_max_pool2d_out_cuda_frame()` kernel**
  - 前向计算kernel，每个线程负责计算一个输出元素
  - 流程：
    1. 根据随机样本获取池化窗口的H和W起始位置
    2. 在该窗口内遍历所有元素，找到最大值
    3. 特殊处理：`poolSizeW < 2 || poolSizeW > 7` 时用不同的循环方式（缓存优化）
    4. 支持NaN值处理（偏向第一个最大值）
    5. 存储最大值和其索引（用于反向传播）

- **`fractional_max_pool2d_out_cuda()` wrapper**
  - 配置grid和block维度
  - 处理3D/4D张量的维度转换
  - 支持Half和BFloat16数据类型

**关键特点：**

- **并行策略**：grid维度 = (输出平面大小/128, 通道数, 批次数)；block维度 = min(128, 输出平面大小)
- **内存访问优化**：针对小和大的poolSizeW采用不同的循环模式
- **索引存储**：使用线性索引 `h * input.size(3) + w` 存储最大值位置
