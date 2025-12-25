这个文件实现了 PyTorch 中的二维最大池化（Max Pooling 2D）操作，支持扩张（dilation）参数。

## 主要功能分解

**元函数部分（at::meta 命名空间）**

- `TORCH_META_FUNC(max_pool2d_with_indices)` 负责参数验证和输出形状推导
  - 验证 kernel_size、stride、padding、dilation 参数的合法性
  - 支持 3D 和 4D 输入张量（带或不带 batch 维度）
  - 支持两种内存格式：ChannelsLast 和 Contiguous
  - 计算输出的高和宽，使用 `pooling_output_shape` 函数
  - 分配两个输出张量：输出值和对应的索引位置

- `TORCH_META_FUNC(max_pool2d_with_indices_backward)` 处理反向传播的形状推导
  - 验证梯度输出与输入的数据类型一致
  - 执行形状检查确保反向传播的兼容性

**实现部分（at::native 命名空间）**

- `TORCH_IMPL_FUNC(max_pool2d_with_indices_out_cpu)` CPU 前向计算
  - 解析并转换所有参数（kernel size、stride、padding、dilation）
  - 调用 `max_pool2d_kernel` 执行实际的池化计算

- `TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_cpu)` CPU 反向计算
  - 梯度输入初始化为零
  - 调用 `max_pool2d_backward_kernel` 进行反向传播

## 关键要点

- **输出格式**：返回两个张量 —— 池化后的值和对应的最大值索引位置
- **参数灵活性**：支持标量或 2-元组形式的 kernel_size、stride、padding、dilation
- **内存布局支持**：处理 ChannelsLast（NHWC）和 Contiguous（NCHW）两种格式
- **分离架构**：通过 DEFINE_DISPATCH 宏分离 CPU/GPU 实现的具体算法细节
