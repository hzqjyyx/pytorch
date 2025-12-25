**Col2Im.cu 文件分析**

这个文件实现了 CUDA 版本的 col2im（column to image）操作，是图像处理中的重要算法。

**核心功能：**

col2im 是 im2col 的逆操作。它将提取的图像块（columns）重新组织回原始图像格式。在深度学习中，这常用于：
- 卷积操作的反向传播
- 反卷积（transposed convolution）实现
- 图像重构任务

**主要流程：**

1. **参数验证（第 26-86 行）**：验证所有输入参数的维度都是 2，包括 output_size、kernel_size、dilation、padding、stride

2. **张量准备（第 88-104 行）**：
   - 确保输入张量连续存储
   - 处理 2D 输入（非批量）by 添加 batch 维度
   - 计算输出张量形状：`[batch_size, output_channels, height, width]`
   - `n_output_plane = n_input_plane / (kernel_width * kernel_height)`

3. **核心计算（第 105-137 行）**：
   - 计算滑动窗口的尺寸（height_col, width_col）
   - 调用 `col2im_batched()` CUDA kernel 进行实际计算
   - 支持多种数据类型（浮点、复数、半精度、BF16、布尔）

4. **输出处理（第 138-140 行）**：如果输入原本是 2D，移除添加的 batch 维度

**接口函数：**

- `col2im_out_cuda_template()`：模板函数，执行核心逻辑
- `col2im_out_cuda()`：写入现有张量
- `col2im_cuda()`：创建新输出张量

**关键特性：**

- 支持非方形卷积核（kernel_height ≠ kernel_width）
- 支持 dilation、padding、stride 参数
- 批量处理（batched）
- GPU 内存检查和验证
