# ConvolutionMM2d.cu 主要功能

这个文件实现了基于**矩阵乘法（Matrix Multiplication）的 2D 卷积**的 CUDA 版本，是一个"慢速"的卷积实现（相对于 cuDNN 等优化库）。

## 核心思路

将卷积运算转换为矩阵乘法（GEMM）来计算，利用高度优化的 BLAS 库：

1. **im2col 变换**：将输入特征图转换为列矩阵
   - 每一列对应一个输出位置的感受野
   - 形状：`[nInputPlane × kH × kW, outputHeight × outputWidth]`

2. **矩阵乘法**：`output = weight × columns`
   - weight 形状：`[nOutputPlane, nInputPlane × kH × kW]`
   - 调用 `at::cuda::blas::gemm` 进行计算

3. **优化**：当卷积核为 1×1 且无 padding、stride=1 时，跳过 im2col 直接做矩阵乘法

## 关键函数

### `slow_conv2d_forward` (aten/src/ATen/native/cuda/ConvolutionMM2d.cu:125-214)
```
输入 → [可选] im2col → GEMM → 输出
```
- 逐 batch 处理
- 如果有 bias，先将 bias 广播到输出
- 对每个样本：提取 columns → GEMM 累加到输出

### `slow_conv2d_shape_check` (aten/src/ATen/native/cuda/ConvolutionMM2d.cu:24-114)
验证：
- 卷积核、stride 参数有效性
- 输入输出维度匹配
- padding 后的尺寸足够容纳卷积核

### `new_view_weight_MM2d` (aten/src/ATen/native/cuda/ConvolutionMM2d.cu:116-123)
将 4D 权重 `[nOut, nIn, kH, kW]` reshape 为 2D 矩阵 `[nOut, nIn×kH×kW]`

## 数据流（Forward）

```
Input [N, C_in, H, W]
  ↓ (逐 batch)
  ↓ im2col → columns [C_in×kH×kW, H_out×W_out]
  ↓ 
Weight [C_out, C_in×kH×kW] × columns
  ↓ GEMM
Output [N, C_out, H_out, W_out]
```

## 性能特点

- **优点**：实现简单，利用成熟的 GEMM 实现
- **缺点**：
  - im2col 需要额外内存（`O(C_in × kH × kW × H_out × W_out)`）
  - 数据重排开销
  - 没有利用卷积的局部性优化

---

**Backward 相关**（简要）：
- `slow_conv2d_backward`：计算输入梯度，通过 GEMM + col2im
- `slow_conv2d_grad_weight`：计算权重梯度，通过 im2col + GEMM
- `slow_conv2d_backward_out_cuda`：统一接口，按需计算 grad_input/grad_weight/grad_bias

**ROCm**：无相关内容（文件仅针对 CUDA）
