这个文件实现了 CUDA 上的 2D 深度可分离卷积(Depthwise Separable Convolution)的前向和反向传播。

## 核心概念

深度可分离卷积的特点是每个输入通道独立进行卷积，然后可选地用 depthwise multiplier 扩展输出通道数。与标准卷积不同，这里的 weight 形状是 `(output_channels, 1, kH, kW)`，其中第二维为 1 表示每个滤波器只作用于一个输入通道。

## 前向传播实现

### 通用 Kernel (conv_depthwise2d_forward_kernel_generic)
aten/src/ATen/native/cuda/DepthwiseConv2d.cu:37-138

处理任意 kernel size 的情况：

1. **索引计算优化**：用除法和乘法替代模运算，从线性索引 linearIndex 计算出 (n, c, h, w) 坐标
2. **边界预计算**：在主循环前计算有效的 kernel 范围 [kHmin, kHmax) 和 [kWmin, kWmax)，避免循环内重复检查边界条件
3. **卷积计算**：只遍历有效范围内的 kernel 位置，累加 weight × input 的结果

### 特化 Kernel (conv_depthwise2d_forward_kernel)
aten/src/ATen/native/cuda/DepthwiseConv2d.cu:145-213

针对常见 kernel size (1×1, 3×3, 5×5) 的优化版本：

1. **模板参数 kSize**：编译时确定循环边界，启用循环展开 (#pragma unroll)
2. **边界检查方式**：不预计算边界，而是在循环内直接判断 `(h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth)`
3. **性能权衡**：小 kernel 情况下，循环展开带来的收益超过边界检查的开销

### 宿主函数 (conv_depthwise2d_forward_out)
aten/src/ATen/native/cuda/DepthwiseConv2d.cu:360-451

1. **输入验证**：检查张量维度、形状匹配、output_channels 是 input_channels 的倍数
2. **Kernel 选择**：根据 kernel size 选择特化版本或通用版本
   - kW=5 且 kH=5 → 特化版本 `<5>`
   - kW=3 且 kH=3 → 特化版本 `<3>`
   - kW=1 且 kH=1 → 特化版本 `<1>`
   - 其他 → 通用版本 `generic`
3. **类型分发**：使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持 float/double/half/bfloat16

## 辅助功能

### dummy_packed_accessor32
aten/src/ATen/native/cuda/DepthwiseConv2d.cu:26-30

当 bias 未定义时，创建一个空的 PackedTensorAccessor，避免 kernel 中的条件分支。

### 公共接口
- `conv_depthwise2d_cuda`: 创建输出张量并调用 `conv_depthwise2d_cuda_out`
- `conv_depthwise2d_cuda_out`: 处理参数并调用底层实现

## ROCm 和 Backward 相关（简要）

- **ROCm 差异**：禁用某些编译器指令如 `C10_LAUNCH_BOUNDS_1` 和 `#pragma unroll`
- **Backward Input Kernel** (conv_depthwise2d_backward_kernel): 计算输入梯度，需要遍历 depthwiseMultiplier 并处理 stride 对齐
- **Backward Weight Kernel** (conv_depthwise2d_grad_weight_kernel): 每个 block 计算一个 weight 元素的梯度，使用 warp-level 并行和 block reduction
- **Stride 特化**：backward kernel 针对 stride=1 和 stride=2 有特化版本以提升性能
