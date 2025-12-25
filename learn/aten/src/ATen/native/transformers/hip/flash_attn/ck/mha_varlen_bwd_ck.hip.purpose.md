这个文件实现了变长序列(variable-length)的多头注意力(Multi-Head Attention)反向传播，采用 AMD ROCm 的 Composable Kernel (CK) 后端。

## 核心功能

**参数准备函数**

`get_ck_fmha_varlen_bwd_traits` 构建反向传播的特征配置：
- 设置 `is_group_mode=true` 表示变长模式
- 配置 mask 类型（causal/local/no_mask）
- 处理 bias、dropout、deterministic 等选项
- 决定是否需要计算 bias 梯度

`get_ck_fmha_varlen_bwd_args` 准备所有张量参数和步幅(stride)信息：
- 输入张量：q, k, v, out, softmax_lse, dout
- 输出张量：dq, dk, dv, d(softmax_d), dq_acc(累加缓冲), 可选的 dbias
- 对每个张量提取三个维度的步幅：batch_stride, stride, nhead_stride
- 变长模式下 batch_stride 都设为 0，因为数据是连续存储的
- seqlens_q/seqlens_k 提供每个样本的实际序列长度边界

**主函数 `mha_varlen_bwd_ck`**

执行流程：
1. **输入验证**：检查数据类型(fp16/bf16)、设备一致性、张量连续性
2. **Mask 配置**：
   - `is_causal=true` → 设为因果mask（`window_size_right=0`）
   - `window_size_left/right=-1` → 无mask
   - 其他情况 → local mask
3. **梯度张量分配**：
   - 若未提供 dq/dk/dv，则创建与输入同形状的空张量
   - dout 需 padding 到 8 的倍数
4. **工作空间分配**：
   - `softmax_d`：存储 softmax 中间梯度，形状 `[batch, num_heads, max_seqlen_q]`
   - `dq_accum`：累加缓冲
     - 非确定性模式：`[1, total_q, num_heads, head_size]`
     - 确定性模式：`[nsplits, total_q, num_heads, head_size]`，需多次分块累加
   - MQA/GQA 需要 `dk_expanded/dv_expanded`：`[total_k, num_heads, head_size]`
5. **CK 反向传播调用**：
   - `fmha_bwd(traits, args, stream_config)` 执行实际计算
   - 使用原子操作更新 dq，因此预先 zero_
6. **后处理**：
   - MQA/GQA：对 dk_expanded/dv_expanded 按组求和 → dk/dv
   - 移除 head_size 的 padding

**内存布局**
- 输入：`q/k/v: [total_tokens, num_heads, head_size]`，total_tokens = sum(seqlens)
- seqlens：`[batch+1]` 累积和数组，标记每个样本边界
- 所有张量最后一维必须连续（stride=1）

**特殊处理**
- `max_seqlen_q=0`：空序列，直接将输出张量置零
- dropout：通过 philox_seed/offset 复现前向的随机mask
- bias 梯度：仅在 `bias_requires_grad=true` 时计算并返回

---

**ROCm/Backward 相关要点：**
- HIP 后端通过 `HIPGuardMasqueradingAsCUDA` 兼容 CUDA 接口
- 使用 Composable Kernel 库的 fmha_bwd 实现
- 确定性模式通过分块计算避免原子操作的不确定性
- 支持 MQA/GQA 优化（num_heads_k < num_heads）
