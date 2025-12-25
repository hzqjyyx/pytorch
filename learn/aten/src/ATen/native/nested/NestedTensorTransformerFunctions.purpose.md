# NestedTensorTransformerFunctions 核心功能

这些文件实现了 **NestedTensor 在 Transformer 模型中的专用优化操作**。NestedTensor 是 PyTorch 用于高效处理变长序列批次的数据结构，避免了传统方法中的 padding 浪费。

## 核心概念

**NestedTensor 结构**：
- 将多个不同形状的张量打包成一个连续的 buffer
- 维护一个 sizes 张量记录每个子张量的形状
- 例如：batch 中有 3 个序列，长度分别为 [5, 3, 7]，NestedTensor 将它们存储为一个形状 `[15, hidden_dim]` 的 buffer

## 主要功能实现

### 1. 矩阵运算优化

**nested_linear** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:58-73)
```cpp
// NT: [batch, var_seq_len, in_features] × W: [out_features, in_features]
// → NT: [batch, var_seq_len, out_features]
```
- 将 NestedTensor 的 buffer reshape 成 2D：`[-1, in_features]`
- 调用标准 `at::linear` 进行批量计算
- 更新输出的 sizes 张量，最后一维改为 `out_features`

**NestedTensor_matmul** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:75-87)
```cpp
// NT × dense_matrix
```
- 类似 linear，但使用 `at::mm` 而非 `at::linear`（无 bias）
- reshape buffer → 矩阵乘法 → 重新包装为 NestedTensor

**NestedTensor_times_Tensor_plus_Tensor_addmm** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:89-122)
```cpp
// alpha * NT × mat2 + beta * self
```
- 实现 `addmm` 操作：`output = beta * self + alpha * (NT × mat2)`
- 支持可选的 GELU 激活融合（通过 `use_gelu` 参数调用 `_addmm_activation`）
- 用于 Transformer 的 FFN 层优化

### 2. 元素级操作

**NestedTensor_add_NestedTensor_in_place** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:124-140)
- 两个 NestedTensor 的就地加法
- 检查 sizes 是否完全匹配
- 直接在 buffer 上执行 `add_` 操作

### 3. Attention 相关

**NestedTensor_softmax_dropout** (CPU版本, aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:142-183)
- 输入：`self` 是 `[batch, num_heads, max_seq, max_seq]` 的 padded 张量
- 对每个序列的有效区域（`[seq_len, seq_len]`）应用 softmax
- 将 padding 区域置零（超出实际序列长度的部分）
- 逐样本循环处理，效率较低但实现简单

**NestedTensor_softmax_dropout_cuda** (CUDA版本, aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:185-191)
- 将 NestedTensor 转换为 mask：`NestedTensor_to_mask`
- 调用融合的 `_masked_softmax` kernel
- 避免 CPU 版本的循环，利用 GPU 并行

### 4. 格式转换工具

**NestedTensor_to_mask** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:213-246)
```cpp
// NT: [batch, var_seq_len, hidden] → mask: [batch, max_seq_len]
```
- 生成布尔 mask，标记哪些位置是 padding（`True`）、哪些是有效数据（`False`）
- 当前只支持 `mask_dim=2` 的 3D NestedTensor
- 遍历每个样本的实际长度，将有效位置设为 `False`

**NestedTensor_batch_offsets_from_size_tensor** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:193-210)
```cpp
// sizes: [batch, ndim] → offsets: [batch+1]
```
- 计算每个子张量在 buffer 中的偏移量
- 例如：sizes = [[2,3], [1,4], [3,2]] → offsets = [0, 6, 10, 16]
- 用于高效索引 NestedTensor 的 buffer

### 5. Jagged ↔ Padded 转换

**_jagged_to_padded_dense_forward_cpu** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:248-288)
```cpp
// values: [total_length, ...] + offsets → padded: [batch, max_length, ...]
```
- 将压缩存储的 jagged tensor 转换为 padded dense tensor
- 使用 offsets 确定每个样本的起止位置
- 超过 `max_length` 的部分会被截断

**_padded_dense_to_jagged_forward_cpu** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:290-345)
```cpp
// padded: [batch, max_length, ...] + offsets → values: [total_length, ...]
```
- 反向操作：从 padded tensor 提取有效数据
- 根据 offsets 确定每个样本的实际长度
- 检查长度不超过 padded 维度

## 约束检查

**check_nested_tensor_matrix_constraints** (aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp:15-55)
- 验证输入必须是 3D NestedTensor（batch × seq × feature）
- dense matrix 必须是 2D
- NestedTensor 必须连续存储
- 最后一维必须与 dense matrix 对应维度匹配
- 不支持 nested weight（weight 必须是普通 Tensor）

## 设计限制

文件头注释明确说明：这些函数 **仅支持 Transformer 特定场景**，不是通用的 NestedTensor 操作：
- 要求 NestedTensor 必须 contiguous
- 固定维度要求（3D NestedTensor × 2D dense）
- 假设只有序列长度维度是可变的，特征维度必须一致

---

**ROCm 相关**：未见明确的 ROCm 特定代码分支

**Backward 相关**：
- 这些是 forward 函数，backward 实现在其他文件（如 `NestedTensorBackward.cpp` 或通过 autograd 自动派生）
- `_jagged_to_padded_dense_forward_cpu` / `_padded_dense_to_jagged_forward_cpu` 的命名暗示存在对应的 backward 版本
