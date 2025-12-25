# ConvUtils.h 核心功能分析

这是 PyTorch 卷积操作的工具头文件，提供了卷积实现选择、参数验证和形状计算的核心功能。

## 1. 后端选择机制 (Backend Selection)

**ConvBackend 枚举** (lines 96-120)：定义了所有可用的卷积后端
- 硬件加速库：`Cudnn`, `Mkldnn`, `Mps`
- CPU 实现：`Slow2d`, `Slow3d`, `SlowDilated2d/3d`, `SlowTranspose2d/3d`
- 特殊优化：`Winograd3x3Depthwise`, `Xnnpack2d`, `NnpackSpatial`
- 空张量处理：`Empty`, `MkldnnEmpty`

**select_conv_backend()** (line 124-127)：根据输入张量属性自动选择最优后端
```cpp
ConvBackend select_conv_backend(
    const Tensor& input, const Tensor& weight, bias,
    stride, padding, dilation, transposed, output_padding, groups, bias_sizes)
```

## 2. 内存格式优化 (Memory Format)

各后端的 Channels Last 支持判断：

**cudnn_conv_suggest_memory_format()** (lines 320-349)
- cuDNN >= 7603：支持 NHWC (ChannelsLast) 用于 2D 卷积
- cuDNN >= 8005：支持 NDHWC (ChannelsLast3d) 用于 3D 卷积
- 禁用 float64 的 Channels Last

**mkldnn_conv_use_channels_last()** (lines 383-408)
- 支持 2D/3D Channels Last
- 排除 float64 和 MkldnnCPU 张量

**thnn_conv_use_channels_last()** (lines 410-420)
- CPU 设备的 2D Channels Last 支持

**xpu_conv_use_channels_last()** (lines 422-438)
- Intel XPU 的 Channels Last 支持

## 3. 形状计算 (Shape Calculation)

### 前向输出形状
**conv_output_size()** (lines 217-248)
```cpp
output_size[d] = (input_size[d] + 2*padding - kernel) / stride + 1
其中 kernel = dilation * (weight_size[d] - 1) + 1
```

### 反向输入形状（转置卷积）
**conv_input_size()** (lines 251-281)
```cpp
input_size[d] = (output_size[d] - 1) * stride - 2*padding + kernel + output_padding
```

### 权重形状推导
**conv_weight_size()** (lines 284-312)
- 从输入/输出形状反推所需的卷积核大小

**关键约束**：
- `input_channels = weight_input_channels * groups` (line 202)
- `output_channels = weight_output_channels` (line 227)

## 4. 参数验证 (Validation)

**check_args()** (lines 156-173)
- 检查 padding/stride/dilation 参数数量
- 确保所有值 ≥ 0

**convolution_shape_check()** (lines 191-210)
- 维度范围检查：3D ≤ dim < 6D (line 201)
- 通道数一致性：`input.channels == weight.channels * groups`
- 输入/权重/输出维度必须相同

## 5. 辅助工具

**reshape_bias()** (line 314-318)
- 将 1D bias 重塑为 `[1, C, 1, ..., 1]` 以便广播

**维度常量** (lines 139-147)
```cpp
input_batch_size_dim = 0, input_channels_dim = 1
output_batch_size_dim = 0, output_channels_dim = 1
weight_output_channels_dim = 0, weight_input_channels_dim = 1
max_dim = 3  // 最多支持 3D 空间卷积
```

## 6. cuDNN V8 配置

**cudnnv8_enabled_check_debug()** (lines 80-89)
- 环境变量 `TORCH_CUDNN_V8_API_DISABLED` 控制启用
- 调试模式 `TORCH_CUDNN_V8_API_DEBUG` 打印前 10 次调用信息

**cudnnv8_use_heur_mode_b()** (lines 74-93)
- 环境变量 `TORCH_CUDNN_USE_HEURISTIC_MODE_B` 选择算法搜索模式

---

## ROCm/MIOpen 相关内容（简要）
- `miopen_conv_use_channels_last()`: MIOpen NHWC 支持（需环境变量 `PYTORCH_MIOPEN_SUGGEST_NHWC`）
- Dispatch stubs: `miopen_convolution_backward_stub`, `miopen_depthwise_convolution_backward_stub`

## Backward 相关内容（简要）
- 多种后端的反向传播函数类型定义（lines 13-71）
- Dispatch stubs 声明：`conv_depthwise2d_backward_stub`, `cudnn_convolution_backward_stub`, `mkldnn_convolution_backward_stub` 等
- 支持输入梯度、权重梯度、bias 梯度的计算
