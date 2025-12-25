ConvolutionTBC.cpp 实现了时间维度卷积（Temporal Backward Convolution）的前向和反向传播。

**前向传播 (conv_tbc)**
- 输入形状：(time, batch, in_channels)
- 权重形状：(kernel_width, in_channels, out_channels)
- 偏置形状：(out_channels)
- 通过循环遍历kernel_width的每个位置，对齐后执行矩阵乘法（addmm_）
- 输出形状：(output_time, batch, out_channels)

**反向传播 (conv_tbc_backward)**
- 计算三个梯度张量：
  - dInput：通过 dOutput × W^T 得到
  - dWeight：通过 I^T × dOutput 得到
  - dBias：通过对dOutput求和得到

**关键特性**
- 支持时间维度的动态padding
- 使用real_pad计算实际padding值以处理边界对齐
- 所有操作基于高效的矩阵乘法（GEMM）
- 采用列主序（column-major）矩阵假设优化性能

**核心操作流程**
- 前向：对每个kernel位置计算 output += input × weight + bias
- 反向：同步计算三个梯度，保持计算图一致性
