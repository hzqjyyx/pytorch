## 主要功能

这两个文件是 PyTorch 量化操作库的注册中心，负责将量化相关的运算符注册到 PyTorch 的调度系统中。

### library.h
定义了三个注册函数的声明：
- `register_linear_params()` - 注册线性层参数
- `register_embedding_params()` - 注册嵌入层参数  
- `register_conv_params<int kSpatialDim>()` - 注册卷积层参数的模板函数，支持 2D 和 3D 卷积

### library.cpp
包含两个主要的 `TORCH_LIBRARY` 宏定义块：

#### 1. `TORCH_LIBRARY(quantized, m)` (lines 4-220)
注册 `quantized::` 命名空间下的所有量化操作，包括：

**基础算术运算**
- `add`, `add_relu` 及其标量和输出变体
- `mul`, `mul_relu` 及其标量和输出变体
- `matmul` - 量化矩阵乘法

**卷积操作**
- `conv1d/2d/3d` 及其 ReLU 融合版本
- `conv_transpose1d/2d/3d` - 转置卷积
- `conv2d_add`, `conv2d_add_relu` - 卷积与加法融合
- 动态量化版本：`conv1d/2d/3d_dynamic`
- 预打包操作：`conv1d/2d/3d_prepack` 和 `unpack`
- 参数查询：`stride`, `padding`, `dilation`, `groups` 等

**线性层操作**
- `linear`, `linear_relu` - 静态量化线性层
- `linear_dynamic` - 动态量化
- `linear_dynamic_fp16` - FP16 动态量化
- `linear_leaky_relu`, `linear_tanh` - 融合激活函数
- `linear_with_input_q_dq_qweight_dq_output_fp32` - 特殊融合模式
- 预打包：`linear_prepack`, `linear_unpack`

**归一化操作**
- `batch_norm`, `batch_norm1d/2d/3d` 及其 ReLU 融合版本
- `group_norm`, `instance_norm`, `layer_norm`

**嵌入操作**
- `embedding_byte`, `embedding_4bit` - 低比特嵌入
- `embedding_bag_byte/4bit/2bit` - 嵌入袋
- `embedding_bag_*_rowwise_offsets` - 行级量化嵌入袋
- 预打包：`embedding_bag_*_prepack/unpack`

**激活函数**
- `relu6`, `leaky_relu`, `prelu`, `sigmoid`, `softmax`
- `elu`, `celu`, `hardswish`, `dropout`
- `clamp`, `threshold`

**池化操作**
- `max_pool1d`, `max_pool2d`

**其他操作**
- `cat`, `cat_relu` - 张量拼接

#### 2. `TORCH_LIBRARY(_quantized, m)` (lines 225-249)
注册 `_quantized::` 命名空间下的操作，这些是移动端使用的内部 API，包括简化版本的：
- `add`, `conv2d/3d`, `conv_transpose1d/2d/3d`
- `linear`, `linear_dynamic`
- FBGEMM 包装函数：`wrapped_fbgemm_pack_gemm_matrix_fp16` 等

#### 3. `TORCH_LIBRARY(onednn, m)` (lines 251-279)
注册 oneDNN 后端的量化操作，用于 PyTorch 2.0 的导出功能：
- `qconv_prepack` - 卷积权重预打包
- `qconv1d/2d/3d_pointwise` - 卷积与逐点操作融合
- `qconv2d_pointwise.binary` - 卷积与二元操作融合
- `qlinear_prepack` - 线性层预打包
- `qlinear_pointwise` - 线性层与逐点操作融合
- `linear_dynamic_fp16` - FP16 动态线性层

### 特点
- 所有操作都标记为 `{at::Tag::pt2_compliant_tag}`，表示兼容 PyTorch 2.0
- 使用 `TORCH_SELECTIVE_SCHEMA` 支持选择性编译
- 包含大量向后兼容的废弃函数（deprecated functions）
- 支持多种融合模式以优化性能（如 conv+relu, linear+relu 等）

---

**ROCm 相关**: 未涉及

**Backward 相关**: 未涉及（量化操作通常用于推理，不需要反向传播）
