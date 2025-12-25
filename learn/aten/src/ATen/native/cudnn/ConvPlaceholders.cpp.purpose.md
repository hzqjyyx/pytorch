# ConvPlaceholders.cpp 文件分析

这个文件是一个条件编译的 **placeholder 实现**，当 PyTorch 编译时 **未启用 cuDNN 支持** 时，提供占位符函数。

## 核心功能

- **条件编译守卫**：`#if !AT_CUDNN_ENABLED()` 包装所有函数，确保只在禁用 cuDNN 时编译
- **错误提示**：所有函数都调用 `TORCH_CHECK(false, ...)` 抛出信息清晰的错误，告诉用户该功能不可用

## 提供的 Placeholder 函数

### Forward 操作
- `cudnn_convolution()` - 标准卷积前向传播
- `cudnn_convolution_out()` - 带输出张量的卷积
- `cudnn_convolution_transpose()` - 转置卷积前向传播
- `raw_cudnn_convolution_forward_out()` - 原始卷积前向（低级 API）

### Activation Fusion
- `cudnn_convolution_relu()` - 卷积 + ReLU 融合
- `cudnn_convolution_add_relu()` - 卷积 + 加法 + ReLU 融合

## 设计意义

- **接口一致性**：即使 cuDNN 不可用，代码也能编译通过，保持 API 接口
- **清晰的错误消息**：用户尝试使用 cuDNN 函数时，获得明确的编译失败提示而非链接错误
- **模块化**：将非 cuDNN 环境下的实现独立到此文件，主实现文件可专注 cuDNN 逻辑
