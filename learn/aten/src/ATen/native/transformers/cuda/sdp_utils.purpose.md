## SDPA Backend Selection - Runtime Dispatch System

这是 PyTorch 中 Scaled Dot Product Attention (SDPA) 的**运行时后端选择系统**，用于在多个优化内核之间自动选择最合适的实现。

### 核心架构

**Backend Priority System** (`select_sdp_backend`)
- 遍历优先级队列：`cudnn_attention` → `flash_attention` → `efficient_attention` → `math`
- 对每个 backend 调用 `can_use_*` 检查函数
- 返回第一个满足所有约束的 backend
- 如果全部失败，开启 debug 模式重新检查并输出详细失败原因

**Filter Chain Pattern**
每个 `can_use_*` 函数维护一组约束检查器（filters）：
```cpp
constexpr auto general_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
    check_runtime_disabled_flash,
    check_all_tensors_on_device,
    check_tensor_shapes,
    // ... 更多检查
);
```
所有 filter 必须通过才能使用该 backend。

---

### Backend-Specific Constraints

#### 1. Flash Attention (`can_use_flash_attention`)

**硬件要求**
- CUDA: SM 8.0 - 12.0 (`check_flash_attention_hardware_support`)
- 数据类型: FP16/BF16（SM 8.0+），FP16（SM 7.0+）

**Head Dimension 限制**
- Q/K/V 最后一维必须相同且 ≤ 256
- Nested Tensor: 必须是 8 的倍数
- SM 8.6/8.9/12.0 特殊限制：
  - 训练模式下 head_dim ∈ (192, 224] 不支持
  - head_dim > 224 时不支持 dropout

**Causal Attention 限制**
- 不支持非方阵的 causal mask（`seqlen_q != seqlen_k`）
- 原因：FlashAttention v2 的 causal mask 对齐方式变更会导致 BC break

**其他约束**
- 不支持显式 attention mask（`check_for_attn_mask`）
- 支持 Grouped Query Attention
- 最后一维必须连续（stride=1）

#### 2. Memory Efficient Attention (`can_use_mem_efficient_attention`)

**硬件要求**
- SM 5.0 - 12.0
- SM < 8.0: FP16/FP32
- SM ≥ 8.0: FP16/FP32/BF16

**Head Dimension 对齐**
- Q/K 最后一维相同且必须是 `minimum_gemm_alignment` 的倍数
- V 最后一维必须是该对齐值的倍数
- 对齐值计算：
  ```cpp
  // Tensor Core: max(4, 128/bits_per_scalar)
  // 非 Tensor Core: SM8.0+ 为 4，否则为 1
  ```

**限制**
- 不支持 Nested Tensor 的训练（`check_requires_grad_and_nested`）
- 不支持 Grouped Query Attention
- 必须最后一维连续（不忽略单例维度）

#### 3. cuDNN Attention (`can_use_cudnn_attention`)

**版本要求**
- cuDNN ≥ 8.9.3（forward）
- cuDNN ≥ 8.9.6（dropout 支持）
- cuDNN ≥ 9.0.0（seq_q < 64 支持）

**Shape 约束**
- head_dim ≤ 128 且必须是 8 的倍数
- seq_len 不能为 1
- cuDNN < 8.9.6: seq_k 必须是 64 的倍数
- cuDNN < 9.0.0: seq_q ≥ 64

**Layout 要求** (`check_cudnn_layout`)
支持两种布局之一：
1. **Packed QKV**: `stride(0)=s*3*h*d, stride(1)=d, stride(2)=3*h*d, stride(3)=1`
2. **Unpacked**: `stride(0)=s*h*d, stride(1)=d, stride(2)=h*d, stride(3)=1`

**其他**
- SM 8.0 - 12.0
- FP16/BF16
- 非确定性算法（deterministic mode 下失败）
- Nested Tensor 需要环境变量 `TORCH_CUDNN_SDPA_NESTED_TENSOR_ENABLED=1` 且仅 SM 9.0 支持

---

### 通用约束检查器

**Tensor 基础检查**
- `check_all_tensors_on_device`: 必须在 CUDA 设备上
- `check_tensor_shapes`: 验证 Q/K/V 的维度匹配
- `check_nonzero_sequence_lengths_dense`: 序列长度 > 0
- `check_last_dim_stride_equals_1_dense`: 最后一维连续性

**Nested Tensor 专用**
- `check_batch_size_nested`: batch size 检查
- `check_for_seq_len_0_nested_tensor`: 序列长度不为 0
- `check_for_seq_len_1_nested_tensor`: 序列长度不为 1（fused kernel 限制）

**Runtime 开关**
- `check_runtime_disabled_*`: 检查全局上下文中用户是否禁用该 backend

---

### 辅助功能

**优先级配置**
```cpp
std::array<SDPBackend, num_backends> priority_order(sdp_params const& params) {
  return at::globalContext().sDPPriorityOrder();
}
```
从全局上下文读取用户自定义优先级。

**Tensor Core 决策**
```cpp
bool use_tensor_cores(sdp_params const& params, cudaDeviceProp* dprops, bool is_half) {
  if (dprops->major >= 8) return true;
  if (dprops->major >= 7) return is_half;
  return false;
}
```

**Debug 模式**
- 所有 filter 都接受 `bool debug` 参数
- debug=true 时打印详细失败原因（TORCH_WARN）
- `select_sdp_backend` 在所有 backend 失败后自动启用 debug 重新检查

---

### ROCm 相关（简要）
- Flash/MemEfficient 共享 AOTriton 后端
- 需要 `aotriton::v2::flash::check_gpu` 检查 GPU 支持
- 实验性架构需设置 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
- MemEfficient 要求 attention mask 为 boolean 或与 Q 同类型

### Backward 相关（简要）
- cuDNN Nested Tensor 不支持 backward
- MemEfficient Nested Tensor 不支持 backward（非 ROCm）
- Flash Attention SM 8.6/8.9 对 head_dim > 192 的训练模式有限制
