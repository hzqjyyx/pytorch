# layer_norm.cpp/h 主要功能

这两个文件实现了 PyTorch 中的 **Layer Normalization** 和 **RMS Normalization** 操作。

## 核心架构

采用 **dispatcher 模式**：
- `.h` 文件声明接口和 dispatch stubs
- `.cpp` 文件提供 CPU 实现和通用入口
- 通过 `DECLARE_DISPATCH`/`DEFINE_DISPATCH` 将计算内核分发到不同后端（CPU/CUDA等）

## Layer Normalization 实现

### 1. 输入验证 (`_check_layer_norm_inputs`)
- 检查 `normalized_shape` 至少是 1 维
- 验证 weight/bias 形状与 `normalized_shape` 匹配
- 验证输入张量的后 n 维与 `normalized_shape` 一致
- 计算并返回 `(M, N)` 参数：
  - `M`: 要归一化的样本数（前面维度的乘积）
  - `N`: 每个样本的特征数（归一化维度的乘积）

### 2. 前向传播路径

**`layer_norm_cpu`** (aten/src/ATen/native/layer_norm.cpp:81):
- 处理 optional 的 weight/bias 参数
- 支持混合数据类型 (mixed_type)
- 调用 `LayerNormKernel` 执行实际计算
- 返回三元组：`(output, mean, rstd)`

**`layer_norm_with_mean_rstd_out`** (aten/src/ATen/native/layer_norm.cpp:40):
- 调用内核计算归一化结果
- 重塑 mean/rstd 的形状以匹配广播语义（在归一化维度上添加维度 1）

**`layer_norm_cpu_out`** (aten/src/ATen/native/layer_norm.cpp:67):
- 输出版本，不计算统计量（mean/rstd）

### 3. Math 实现路径

**`math_native_layer_norm`** (aten/src/ATen/native/layer_norm.cpp:205):
- 来自 pytorch/xla 的移植实现
- 通过 **重用 Batch Normalization** 实现 Layer Normalization：
  1. 将输入重塑为 `(1, M, N)`
  2. 调用 `native_batch_norm` 计算归一化
  3. 手动应用 weight/bias（逐元素而非逐通道）
- 特殊处理零大小输入

### 4. 符号整数支持

**`layer_norm_symint`** (aten/src/ATen/native/layer_norm.cpp:193):
- 支持符号形状（用于编译和导出）
- 简单包装 `native_layer_norm_symint` 返回第一个元素

## RMS Normalization 实现

**`rms_norm_symint`** (aten/src/ATen/native/layer_norm.cpp:264):

计算公式：`output = input / sqrt(mean(input^2) + eps) * weight`

实现细节：
1. **输入验证**：通过 `_check_rms_norm_inputs_symint` 检查形状匹配
2. **MPS 加速**：在 Apple Silicon 上且满足条件时使用 MPS 内核
3. **数据类型调度**：
   - 使用 `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2` 处理多种类型
   - FP16/BF16 自动提升到 FP32/FP64 进行计算（opmath）
4. **计算步骤**：
   - 沿归一化维度计算 `pow(input, 2).mean()`
   - 加 epsilon 后取 `rsqrt`
   - 乘以输入和可选的 weight
   - 转换回原始数据类型

## Dispatch 机制

```cpp
using forward_fn = void (*)(const Tensor& X, const Tensor& gamma, 
                             const Tensor& beta, int64_t M, int64_t N, 
                             double eps, Tensor* Y, Tensor* mean, Tensor* rstd);

DECLARE_DISPATCH(forward_fn, LayerNormKernel)
DEFINE_DISPATCH(LayerNormKernel);
```

- `LayerNormKernel` 在运行时分发到 CPU/CUDA/其他后端的具体实现
- 后端实现在对应的 `cpu/` 或 `cuda/` 目录中

---

**ROCm/Backward 相关**：
- `layer_norm_backward_cpu` (line 118): 计算梯度 `(dX, dgamma, dbeta)`，支持可选梯度 mask
- `LayerNormBackwardKernel`: backward 的 dispatch stub
- 使用 `grad_input_mask` 避免不必要的梯度计算
