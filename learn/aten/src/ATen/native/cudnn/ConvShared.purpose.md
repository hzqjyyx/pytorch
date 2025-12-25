# ConvShared.cpp/h 核心功能分析

这两个文件是 PyTorch 中 cuDNN 卷积操作的**共享层**，提供 cuDNN v7 和 v8 API 共用的核心逻辑。

## 主要组件

### 1. ConvolutionParams 结构体 (ConvShared.h:20-35)
用于卷积参数的哈希计算和缓存的 POD 结构：
- 存储设备 ID、数据类型、输入/权重尺寸
- 记录 padding、stride、dilation、groups 等卷积参数
- 包含 deterministic 和 allow_tf32 标志
- 通过 `setConvolutionParams()` 函数填充，而非构造函数（保持 POD 特性）

### 2. 前向卷积入口 (ConvShared.cpp:240-306)

**`cudnn_convolution()`** - 主要前向接口：
```
input → cudnn_conv_suggest_memory_format() → 确定内存布局
     → conv_output_size() → 计算输出尺寸
     → at::detail::empty_cuda() → 分配输出张量
     → cudnn_convolution_forward_out() → 执行卷积
```

**`cudnn_convolution_forward_out()`** (205-238):
- 检查类型和 GPU 一致性
- 将输入和权重转为连续内存格式
- 调用 `raw_cudnn_convolution_forward_out()`（v7/v8 实现不同）

### 3. 融合操作 (705-815)

**`cudnn_convolution_relu()`** - Conv + ReLU 融合：
- 接受可选 bias 参数
- 使用 `raw_cudnn_convolution_add_relu_out()` 实现
- 将 output_t 用作 z 参数（alpha=0）

**`cudnn_convolution_add_relu()`** - Conv + Add + ReLU 融合：
- 支持残差连接：`output = ReLU(conv(input) + alpha * z + bias)`
- 处理 z 张量的内存格式转换
- 从 globalContext 获取 benchmark 和 allow_tf32 设置

### 4. 辅助功能

**`repro_from_args()`** (139-197):
生成 Python 复现代码片段，用于调试 cuDNN 错误：
```python
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([N,C,H,W], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(in_channels, out_channels, kernel_size=..., padding=..., ...)
out = net(data)
out.backward(torch.randn_like(out))
```

### 5. Raw API 声明 (ConvShared.h:64-128)

声明实际调用 cuDNN 的底层函数（由 Conv_v7.cpp/Conv_v8.cpp 实现）：
- `raw_cudnn_convolution_forward_out`
- `raw_cudnn_convolution_add_relu_out`
- `raw_cudnn_convolution_add_relu_fallback_out`

## 设计要点

**参数检查分层** (ConvShared.cpp:67-72):
- `at::Tensor` 层：TensorArg 分配
- `TensorArg` 层：类型/GPU/形状检查

**内存格式处理**:
- 使用 `cudnn_conv_suggest_memory_format()` 推荐格式
- 支持 ChannelsLast/ChannelsLast3d
- 自动调用 `.contiguous(memory_format)` 确保内存连续

**空张量处理**:
```cpp
if (output_t.numel() == 0) {
    return output_t;  // 直接返回，不调用 cuDNN
}
```

---

## Backward 相关（简要）
- `cudnn_convolution_backward_input()` - 计算输入梯度
- `cudnn_convolution_backward_weight()` - 计算权重梯度
- `cudnn_convolution_backward()` - 根据 output_mask 选择性计算梯度
- `cudnn_convolution_transpose_*()` - 转置卷积（复用 backward_input 逻辑）

## ROCm 相关
- 文件头 `#if AT_CUDNN_ENABLED()` 控制编译
- ConvPlaceholders.cpp 提供 cuDNN 未启用时的占位实现
