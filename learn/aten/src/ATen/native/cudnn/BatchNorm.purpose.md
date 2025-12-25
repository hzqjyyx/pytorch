# BatchNorm.cpp/h 核心功能分析

这些文件实现了 PyTorch 中使用 cuDNN 加速的 Batch Normalization 操作。

## 主要函数

### 1. `cudnn_batch_norm` (前向传播)
**位置**: aten/src/ATen/native/cudnn/BatchNorm.cpp:123-266

**核心流程**:

1. **参数处理与验证** (123-169行)
   - 解包可选 tensor (bias, running_mean, running_var)
   - 检查所有输入是否定义、在同一 GPU、类型匹配
   - 特殊处理: Half 类型输入要求 weight 为 Float
   - 验证 input 维度在 [2, 6) 范围内
   - 确认 weight/bias/running_mean/running_var 的元素数量等于 `num_features`

2. **模式选择** (168-169行)
   ```cpp
   cudnnBatchNormMode_t mode = getCudnnBatchNormMode(
       training, input->suggest_memory_format(), input->dim());
   ```
   根据训练模式、内存格式、维度选择 cuDNN 批归一化模式

3. **训练模式** (189-238行)
   - 分配 `save_mean` 和 `save_var` 用于保存统计信息
   - 获取并分配 workspace 和 reserve space
   - 调用 `cudnnBatchNormalizationForwardTrainingEx`:
     - 使用 `exponential_average_factor` 更新 running statistics
     - 计算并保存 batch mean/var 用于反向传播
     - 输出归一化后的结果

4. **推理模式** (239-258行)
   - 使用预先计算的 `running_mean` 和 `running_var`
   - 调用 `cudnnBatchNormalizationForwardInference`
   - 不需要保存统计信息

**返回值**: `{output, save_mean, save_var, reserve}`

### 2. `_get_cudnn_batch_norm_reserve_space_size`
**位置**: aten/src/ATen/native/cudnn/BatchNorm.cpp:108-121

**功能**: 查询 cuDNN 所需的 reserve space 大小
- 在训练模式下，cuDNN 需要额外内存保存中间状态用于反向传播
- 调用 `cudnnGetBatchNormalizationTrainingExReserveSpaceSize` 获取大小

## 辅助函数

### `expandScale` (79-85行)
```cpp
Tensor expandScale(const Tensor& t, int64_t dim)
```
将 1D 的 weight/bias tensor 扩展为与 input 维度匹配的形状
- 例如: `[C]` → `[1, C, 1, 1]` (对于4D输入)

### `getCudnnBatchNormMode` (87-104行)
选择 cuDNN 批归一化模式:
- **2D 输入**: `CUDNN_BATCHNORM_PER_ACTIVATION` (逐元素归一化)
- **训练 + ChannelsLast/ChannelsLast3d**: `CUDNN_BATCHNORM_SPATIAL_PERSISTENT` (优化的空间归一化)
- **其他情况**: `CUDNN_BATCHNORM_SPATIAL` (标准空间归一化)
  - 注释指出 PERSISTENT 模式在某些模型(ResNeXt-101, R(2+1)D)中可能导致精度损失

## 条件编译逻辑

### 无 cuDNN 支持 (12-54行)
当 `AT_CUDNN_ENABLED()` 为 false 时:
- 所有函数只是抛出错误信息
- 提示 "ATen not compiled with cuDNN support"

### 平台特定头文件 (6-10行)
- **AMD ROCm**: 包含 `ATen/native/cudnn/hip/BatchNorm.h`
- **NVIDIA CUDA**: 包含 `ATen/native/cudnn/BatchNorm.h`

---

## 其他内容 (简要)

**ROCm 支持**:
- 通过 `#ifdef __HIP_PLATFORM_AMD__` 条件编译支持 AMD GPU
- 使用不同的头文件路径

**Backward 相关**:
- `cudnn_batch_norm_backward` (271-393行): 反向传播函数
- 使用 `cudnnBatchNormalizationBackwardEx` 计算梯度
- 返回 `{grad_input, grad_weight, grad_bias}`
- 需要 forward 时保存的 `save_mean`, `save_var`, `reserve` 数据
