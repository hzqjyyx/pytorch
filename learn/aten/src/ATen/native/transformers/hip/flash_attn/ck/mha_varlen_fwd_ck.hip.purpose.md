# mha_varlen_fwd_ck.hip 主要功能

这个文件实现了变长序列的多头注意力（Multi-Head Attention）前向传播，专门针对 AMD ROCm/HIP 平台使用 Composable Kernel (CK) 库。

## 核心函数

### 1. `get_ck_fmha_varlen_fwd_traits()`
配置 FMHA 前向传播的特性参数：
- 设置 head_size、数据类型（fp16/bf16）
- 启用 group mode 和 V 矩阵 row-major 布局
- 配置 mask 类型（causal/local/no_mask）
- 设置是否使用 bias、LSE（log-sum-exp）、dropout

### 2. `get_ck_fmha_varlen_fwd_args()`
准备 FMHA 前向传播的运行时参数：
- **输入张量**：Q (total_q × h × d)、K/V (total_k × h_k × d)
- **序列信息**：cu_seqlens_q/k（累积序列长度，用于变长序列）
- **stride 计算**：batch、nhead、序列维度的 stride
- **可选参数**：attn_bias、dropout_randval、softmax_lse
- **缩放因子**：softmax_scale、dropout 概率

### 3. `mha_varlen_fwd_ck()` - 主入口函数

**输入处理**：
- 验证数据类型（仅支持 fp16/bf16）
- 检查张量形状和连续性
- 解析 batch_size、num_heads、head_size 等维度

**Mask 处理**：
```cpp
// 根据参数选择 mask 类型
if (is_causal) {
    mask = "b:window_left,0"  // causal mask
} else if (no_window) {
    mask = "0"                 // no mask
} else {
    mask = "b:window_left,window_right"  // local/sliding window
}
```

**Head Size Padding**：
- 如果 head_size 不是 8 的倍数，padding 到 8 的倍数
- 提升计算效率（aten/src/ATen/native/transformers/hip/flash_attn/ck/mha_varlen_fwd_ck.hip:243-252）

**输出张量分配**：
- `out`: (total_q, num_heads, head_size) - 注意力输出
- `softmax_lse`: (batch, num_heads, max_seqlen_q) - log-sum-exp 用于梯度计算
- `p`: (num_heads, total_q, max_seqlen_k) - dropout mask（如果需要）

**Dropout 随机数生成**：
```cpp
// 使用 Philox RNG 生成 dropout seed/offset
auto philox_args = gen->philox_cuda_state(counter_offset);
hipLaunchKernelGGL(ParsePhiloxCudaState, ...)  // aten/src/ATen/native/transformers/hip/flash_attn/ck/mha_varlen_fwd_ck.hip:306-307
```

**执行 FMHA Kernel**：
```cpp
float t = fmha_fwd(traits, args, stream_config);  // aten/src/ATen/native/transformers/hip/flash_attn/ck/mha_varlen_fwd_ck.hip:348
```

**后处理**：
- 如果做了 padding，裁剪回原始 head_size
- 处理 seqlen_k == 0 的边界情况（输出置零）

## 变长序列处理机制

使用 **cumulative sequence lengths** (cu_seqlens) 而非固定 batch 维度：
- `cu_seqlens_q/k`: 长度为 (batch+1) 的数组，存储累积序列长度
- 例如 batch=3，序列长度 [5, 3, 7]，则 cu_seqlens = [0, 5, 8, 15]
- 允许不同样本有不同序列长度，避免 padding 浪费

## 特殊优化

1. **GQA 支持**：num_heads % num_heads_k == 0（Grouped Query Attention）
2. **max_seqlen_q == 1 时禁用 causal**：单 token 生成时不需要 mask（aten/src/ATen/native/transformers/hip/flash_attn/ck/mha_varlen_fwd_ck.hip:203）
3. **Window size 自动调整**：窗口大于 max_seqlen 时改为全局注意力（aten/src/ATen/native/transformers/hip/flash_attn/ck/mha_varlen_fwd_ck.hip:216-217）

---

**ROCm/HIP 相关**：
- 使用 `hipLaunchKernelGGL` 启动 kernel
- `HIPGuardMasqueradingAsCUDA` 设备管理
- `getCurrentHIPStreamMasqueradingAsCUDA()` 流管理

**Backward 相关**：
- 返回 `softmax_lse` 用于反向传播梯度计算
- 可选返回 `dropout_randval` 用于梯度中重现 dropout mask
