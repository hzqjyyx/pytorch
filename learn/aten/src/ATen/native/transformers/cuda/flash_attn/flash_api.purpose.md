## Flash Attention API 核心功能

这是 PyTorch 中 Flash Attention 的 C++ API 封装层，负责在 PyTorch 和底层 Flash Attention CUDA 实现之间建立桥梁。

### 主要数据流

```
PyTorch Tensor → 参数验证/转换 → Flash Attention CUDA Kernel → 输出 Tensor
```

### 核心函数：mha_fwd (Multi-Head Attention Forward)

**输入处理**：
- Q/K/V 张量：支持 `[batch, seqlen, num_heads, head_size]` 格式
- 自动处理不同数据类型（FP16/BF16）和布局（连续/非连续内存）
- 支持 Grouped Query Attention (GQA)：允许 K/V 的 head 数量少于 Q

**关键参数转换**：
- `softmax_scale`：默认为 `1/√d`，控制注意力分布的锐度
- `is_causal`：因果掩码，自动设置窗口为 `[∞, 0]`
- `window_size`：滑动窗口注意力，限制每个 token 的可见范围
- `softcap`：Softmax 饱和值上限，防止数值溢出
- `p_dropout`：需要时自动初始化 RNG 状态

**内存优化**：
- 检测非连续张量并创建连续副本（aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp:144-162）
- 复用输入张量作为输出（若提供 `out_`）
- Dropout 使用 Philox RNG，生成固定的种子/偏移量用于反向传播

**调用底层实现**：
```cpp
out = flash::mha_fwd(
    q_padded, k_padded, v_padded,
    softmax_lse, p_dropout, softmax_scale,
    is_causal, window_size_left, window_size_right,
    softcap, return_softmax, gen
);
```

### 核心函数：mha_varlen_fwd (Variable Length Forward)

**适用场景**：批次中序列长度不一致时使用

**关键区别**：
- 输入形状：`[total_tokens, num_heads, head_size]`，其中 `total_tokens = Σ seq_len_i`
- `cu_seqlens_q/k`：累积序列长度索引 `[0, s1, s1+s2, ..., total]`，shape 为 `[batch+1]`
- `seqused_k`：每个样本实际使用的 K/V 长度（支持 padding）
- `block_table`：用于 PagedAttention 的块索引表

**内存管理**：
- 可选 `zero_tensors` 参数：是否清零输出缓冲区
- 支持 paged KV cache（通过 block_table）

### 辅助功能

**张量连续性处理**（flash_api.cpp:24-39）：
```cpp
maybe_make_contiguous() {
    if (is_contiguous(stride_order))
        return tensor;
    return tensor.contiguous();  // 创建副本
}
```

**形状/步长验证**：
- 检查 Q/K/V 维度兼容性
- 验证 head_size ≤ 256（硬件限制）
- 确保 num_heads_k 能整除 num_heads（GQA 要求）

**RNG 状态管理**（flash_api.cpp:248-264）：
- 从 PyTorch Generator 提取 Philox 种子
- 计算正确的偏移量（考虑多线程）
- 返回种子/偏移量供反向传播复用

---

### 其他内容（简要）

**ROCm 平台支持**：
- 编译时通过 `USE_ROCM` 宏区分
- ROCm 使用 CK (Composable Kernel) 后端替代 CUDA 实现
- 参数映射基本一致，但底层调用不同库

**Backward 函数**：
- `mha_bwd` / `mha_varlen_bwd`：计算 dQ, dK, dV 梯度
- 复用 forward 的 softmax_lse 和 RNG 状态
- 支持 `deterministic` 模式（禁用原子加法优化）
- 使用相同的窗口/掩码/dropout 配置保证一致性
