# attention_backward.cu 主要功能

这个文件实现了 PyTorch 中 Scaled Dot-Product Attention 的 CUDA 反向传播核心逻辑，为自注意力机制提供梯度计算支持。

## 核心实现

### 1. Flash Attention 反向传播 (`_flash_attention_backward`)

**入口函数**: aten/src/ATen/native/transformers/cuda/attention_backward.cu:62-169

处理两种前向传播模式的反向计算：

- **变长序列模式** (Varlen): 当 `cumulative_sequence_length_q` 已定义时
  - 调用 `FLASH_NAMESPACE::mha_varlen_bwd` (aten/src/ATen/native/transformers/cuda/attention_backward.cu:116-140)
  - 使用累积序列长度索引处理批次内不同长度的序列

- **密集模式** (Dense): 标准固定长度序列
  - 调用 `FLASH_NAMESPACE::mha_bwd` (aten/src/ATen/native/transformers/cuda/attention_backward.cu:144-163)

关键特性：
- 计算 softmax 缩放因子 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:81)
- 确保梯度输出连续性 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:83-84)
- 支持窗口注意力、dropout、causal masking
- 可选确定性算法 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:100-110)

### 2. cuDNN Attention 反向传播 (`_scaled_dot_product_cudnn_attention_backward_cuda`)

**函数位置**: aten/src/ATen/native/transformers/cuda/attention_backward.cu:171-249

使用 cuDNN 库的高度优化实现：
- 自动扩展 `attn_bias` 到正确维度 (2D/3D/4D → 4D) (aten/src/ATen/native/transformers/cuda/attention_backward.cu:208-221)
- 调用 `run_cudnn_SDP_bprop` 执行实际计算 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:227-247)
- 返回 Q/K/V 的梯度

### 3. Memory Efficient Attention 反向传播 (`_efficient_attention_backward`)

**函数位置**: aten/src/ATen/native/transformers/cuda/attention_backward.cu:251-827

最复杂的实现，针对内存优化的注意力机制：

#### 输入验证 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:292-345)
- 检查张量维度一致性（batch、seqlen、heads、embedding）
- 确保张量连续性和对齐
- 验证 `cu_seqlens` 参数正确性

#### 梯度张量分配 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:357-394)

**共享存储优化**: 当 `shared_storage_dqdkdv=true` 时
```cpp
at::Tensor chunk = at::empty({B, M, 3, nH, K}, query.options());
grad_q = chunk.select(2, 0);
grad_k = chunk.select(2, 1);
grad_v = chunk.select(2, 2);
```
- 单次分配连续内存，避免后续 `torch.cat` 操作
- 要求 Q/K/V 序列长度和嵌入维度相同

#### CUDA 实现核心 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:565-822)

**内核选择机制** (aten/src/ATen/native/transformers/cuda/attention_backward.cu:573-815):
- Lambda `launchKernel` 自动选择最优 CUTLASS 内核
- 验证条件：
  - 最大嵌入维度 `kMaxK` (aten/src/ATen/native/transformers/cuda/attention_backward.cu:582-584)
  - Dropout 支持 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:586-588)
  - 内存对齐 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:595-599)
  - 共享内存限制 (aten/src/ATen/native/transformers/cuda/attention_backward.cu:600-604)

**Delta 计算** (aten/src/ATen/native/transformers/cuda/attention_backward.cu:610-615):
```cpp
auto delta = Kernel::kKernelComputesDelta
    ? at::empty({B, nH, M}, ...)
    : (grad_out * out).sum(-1).transpose(-2, -1).contiguous();
```
- 反向传播中间变量
- 部分内核内部计算，否则预先计算

**分片键优化** (aten/src/ATen/native/transformers/cuda/attention_backward.cu:721-747):
- 自动计算最优 `num_splits_key`
- 平衡并行度和内存使用
- 限制条件：需要累积 gK/gV 时减少分片

### 4. 包装函数

#### `_scaled_dot_product_flash_attention_backward_cuda` (aten/src/ATen/native/transformers/cuda/attention_backward.cu:829-878)
- 转置输入 (BNHD → BNHD)
- 调用 `_flash_attention_backward`
- 转置梯度回原始格式

#### `_scaled_dot_product_efficient_attention_backward_cuda` (aten/src/ATen/native/transformers/cuda/attention_backward.cu:881-942)
- 处理 `grad_input_mask` 控制哪些梯度需要计算
- 转换 causal 标志到 `CustomMaskType`
- 调用 `_efficient_attention_backward`

---

## ROCm 相关实现要点

- 使用 CK (Composable Kernel) 或 AOTriton 后端
- 张量需转置为 (B,H,M,K) 格式
- 变长序列路径调用 `attn_bwd_compact_varlen`
- 密集路径根据头维度选择 `attn_bwd_fused` 或 `attn_bwd`
- 不支持 `num_splits_key` 和 `window_size` 参数

## Backward 函数返回值

- Flash/cuDNN 版本：`(grad_q, grad_k, grad_v)`
- Memory Efficient 版本：`(grad_q, grad_k, grad_v, grad_bias)`
- 所有梯度张量与对应输入形状完全一致
