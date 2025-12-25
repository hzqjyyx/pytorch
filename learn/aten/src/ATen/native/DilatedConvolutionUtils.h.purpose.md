**文件概述**

这是 PyTorch ATen 库中用于扩张卷积（Dilated Convolution）的工具头文件，提供形状检查和输出尺寸计算功能。

**主要功能**

- **宏定义 `TORCH_CHECK_DIM_SIZE`**（第10-20行）
  - 检查张量的维度和特定维度的大小是否符合预期
  - 失败时输出详细的错误信息

- **辅助函数**（第24-32行）
  - `all_positive()`：检查数组中所有元素是否大于0
  - `all_nonnegative()`：检查数组中所有元素是否非负

- **`get_output_size<dim>()`** - 两个重载版本
  - 第一版本（第38-54行）：基于输入张量、kernel_size、stride、pad、dilation计算输出尺寸
  - 第二版本（第58-72行）：同时考虑权重张量，在计算结果前添加output channels和batch size

- **`slow_conv_dilated_shape_check<dim>()`**（第78-227行）
  - 对扩张卷积的所有输入进行完整的形状和维度验证
  - 检查内容包括：
    - 参数数组长度（kernel_size、stride、dilation、pad）必须等于dim
    - 所有参数值必须为正数
    - 输入张量维度必须为 dim+1 或 dim+2（是否batch）
    - 计算的输出尺寸必须非负
    - 权重张量维度必须为 dim+2
    - 权重的kernel部分必须与kernel_size匹配
    - 输入通道数必须与权重匹配
    - bias（如果存在）必须是1D且大小与output channels匹配
    - grad_output（如果存在）尺寸必须与计算值一致

**总结**

| 功能 | 说明 |
|------|------|
| **尺寸计算** | 根据卷积参数计算输出张量尺寸 |
| **参数验证** | 对kernel_size、stride、dilation、pad等参数检查 |
| **张量验证** | 对input、weight、bias、grad_output的维度和大小检查 |
| **模板化设计** | 支持不同维度的卷积（1D、2D、3D等） |
