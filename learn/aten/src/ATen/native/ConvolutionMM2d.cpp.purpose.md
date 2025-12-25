# ConvolutionMM2d.cpp 核心功能

这个文件实现了基于矩阵乘法(MM, Matrix Multiplication)的2D卷积的CPU端forward计算。核心思想是将卷积操作转换为矩阵乘法：

## 主要计算流程

### 1. Im2Col 转换 (`compute_columns2d`)
将输入的4D图像张量转换为2D列矩阵（展开操作）：

- **1x1卷积优化**：当kernel为1x1、stride为1、padding为0时，直接通过view/as_strided操作，避免实际数据拷贝
- **一般情况**：调用 `unfolded2d_copy_stub` 将滑动窗口展开成列
- **输出形状**：
  - Channels-first: `[batch, C_in * kH * kW, outH * outW]`
  - Channels-last: `[batch, outH * outW, kH * kW * C_in]`

### 2. 矩阵乘法 (`slow_conv2d_update_output_frame`)
将卷积转换为 `output = weight * columns`：

- **Channels-first格式**：
  ```
  [outH*outW, C_out] = [outH*outW, C_in*kH*kW] × [C_in*kH*kW, C_out]
  ```
  
- **Channels-last格式**：
  ```
  [outH*outW, C_out] = [outH*outW, kH*kW*C_in] × [kH*kW*C_in, C_out]^T
  ```

- 调用 `cpublas::gemm` 执行实际的矩阵乘法
- 如果有bias，通过设置beta=1复用输出缓冲区中预填充的bias值

### 3. 前向传播主函数 (`slow_conv2d_forward_out_cpu`)

执行流程：
1. 参数验证：调用 `slow_conv2d_shape_check` 检查维度合法性
2. 确定内存格式：根据输入和权重选择ChannelsLast或Contiguous
3. 权重reshape：通过 `view_weight_2d` 将4D权重转为2D (C_out, C_in*kH*kW)
4. Im2Col：将输入展开为列矩阵
5. Bias预填充：如果有bias，先将output填充为broadcast后的bias值
6. 并行计算：对batch维度并行，每个样本执行矩阵乘法

## 关键设计点

**内存格式支持**：
- 同时支持NCHW (Contiguous)和NHWC (ChannelsLast)两种布局
- 根据输入数据格式自动选择最优路径

**并行化**：
- 使用 `at::parallel_for` 在batch维度并行处理
- Forward时每个线程处理一个batch样本

**类型支持**：
- Forward支持所有数值类型 + BFloat16 + Half (`AT_DISPATCH_ALL_TYPES_AND2`)

**性能优化**：
- 1x1卷积的零拷贝优化
- 尽可能复用内存（bias通过gemm的beta参数直接累加）

---

## 其他功能（Backward & ROCm）

- **Backward Input** (`slow_conv2d_backward_update_grad_input_frame`): 通过 weight^T × grad_output 计算输入梯度，然后用 `unfolded2d_acc_stub` 累加回原始输入形状
- **Backward Weight** (`slow_conv2d_backward_weight_frame`): 通过 grad_output × columns^T 累积权重梯度
- **Backward Bias**: 对grad_output在batch/spatial维度求和
- **仅支持浮点类型梯度计算** (`AT_DISPATCH_FLOATING_TYPES_AND2`)
- **ROCm**: 文件中未见ROCm特定代码，仅为纯CPU实现
