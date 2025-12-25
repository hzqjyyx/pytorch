## 文件主要功能

这个文件实现了量化卷积操作的权重解包(unpack)功能，支持多种后端引擎。

### 核心机制

- **运行时多态**：通过 `packed_weight` 指针（类型为 `intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>`）在运行时确定具体的后端实现
- **后端支持**：FBGEMM、QNNPACK、CUDNN、ONEDNN
- **实现位置**：
  - FBGEMM/QNNPACK: `/cpu/qconv_unpack_impl.cpp`
  - CUDNN: `/cudnn/ConvUnpackImpl.cpp`

### 主要类和函数

**QConvUnpackWeightsInt8<kSpatialDim>** (模板类)
- 解包权重张量，返回解包后的权重和可选的偏置
- 支持 Conv2d 和 Conv3d
- 输入形状：`[output_channels, kernel_height, kernel_width, input_channels/Groups]`

**QConv1dUnpackWeightsInt8**
- 专门处理 Conv1d，通过在 Conv2d 基础上挤压维度实现
- QNNPACK 和 ONEDNN 版本需要 clone 权重后再操作

**参数查询类**
- `QConvStride`, `QConvPadding`, `QConvOutputPadding`, `QConvDilation`, `QConvGroups`, `QConvTranspose`
- 从 packed_weight 中提取对应的卷积参数

**unpack_quantized_prepacked_sizes_conv2d**
- 返回权重、偏置大小及卷积参数的元组

### 库注册

在 `TORCH_LIBRARY_IMPL(quantized, CatchAll, m)` 中注册：

- Conv1d/Conv2d/Conv3d 及其 Transpose 版本的 unpack 操作
- Conv2d 的 unpack_sizes 操作（用于查询预打包权重的大小信息）
- Conv2d/Conv3d 及其 Transpose 版本的参数查询操作（stride、padding、dilation 等）

### 关键特性

- 支持 1D、2D、3D 卷积
- 支持转置卷积(ConvTranspose)
- 引擎选择通过全局上下文 `at::globalContext().qEngine()` 确定
- 不支持的引擎会抛出检查错误
