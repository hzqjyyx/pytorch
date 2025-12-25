**主要功能分析：**

- **元信息定义 (Meta Function)**：`TORCH_META_FUNC(adaptive_max_pool2d)` 负责输出张量的形状推导
  - 验证输入为 3D 或 4D 张量
  - 检查非批次维度的大小非零
  - 根据输入维度计算输出形状 (batch_size, channels, output_height, output_width)
  - 设置两个输出：pooling 结果张量和记录最大值索引位置的长整型张量

- **CPU 实现**：`TORCH_IMPL_FUNC(adaptive_max_pool2d_out_cpu)` 调用核心计算函数
  - 调用 `adaptive_max_pool2d_kernel` 执行实际的自适应最大池化运算
  - 将输入自适应池化到指定的输出尺寸，同时记录最大值所在的位置

- **后向传播元信息**：`TORCH_META_FUNC(adaptive_max_pool2d_backward)` 定义梯度反向传播的输出形状
  - 验证梯度张量维度和输入维度匹配
  - 验证数据类型和大小一致性
  - 输出梯度张量形状与原始输入相同

- **后向传播 CPU 实现**：`TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_cpu)` 计算反向梯度
  - 初始化梯度张量为零
  - 调用 `adaptive_max_pool2d_backward_kernel` 根据前向时记录的索引回传梯度

- **调度系统**：通过 `DEFINE_DISPATCH` 宏为两个核心函数注册调度机制，支持多后端实现
