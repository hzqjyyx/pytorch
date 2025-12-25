# mha_fwd_ck.hip 核心功能分析

这个文件实现了 **Multi-Head Attention (MHA) 的前向传播**，专门针对 AMD GPU 使用 Composable Kernel (CK) 库进行优化。

## 主要函数

### 1. `get_ck_fmha_fwd_traits` (L13-30)
配置 FMHA 前向传播的特征参数：
- 输入：mask 信息、数据类型、head size、dropout/lse/bias 开关
- 输出：`fmha_fwd_traits` 结构体
- 关键配置：
  - `is_v_rowmajor = true`：V 矩阵按行主序存储
  - `is_group_mode = false`：非分组模式
  - 根据 `enable_bias` 决定是否使用 elementwise bias

### 2. `get_ck_fmha_fwd_args` (L32-145)
准备 FMHA 前向传播的运行时参数：
- **输入张量及其 stride 计算**：
  - Q/K/V 的 batch、head、sequence 三个维度的 stride
  - 可选的 attention bias 张量处理 (L88-96)
  - dropout randval 和 softmax_lse 的 stride
- **返回 `fmha_fwd_args` 结构体**，包含所有数据指针和维度信息

### 3. `mha_fwd_ck` (L147-365) - **主入口函数**

完整的 MHA 前向传播流程：

#### 输入验证 (L161-187)
- 数据类型检查：仅支持 fp16/bf16
- 张量连续性检查：最后一维必须连续
- 维度约束：`head_size ≤ 256` 且是 8 的倍数
- MQA/GQA 支持：`num_heads % num_heads_k == 0`

#### Mask 配置 (L189-209)
根据参数构造 mask_info：
- **Causal mask**: `is_causal=true` → `window_size_right=0`
- **No mask**: 两个 window size 都是 -1
- **Local attention**: 通用的滑动窗口 mask

#### 优化：seqlenq_ngroups_swapped (L213-220)
当满足以下条件时，重排 Q 的维度以提升性能：
- `seqlen_q == 1` (推理场景)
- `num_heads > num_heads_k` (MQA/GQA)
- 无窗口限制、无 dropout、无 bias
- 将 `(b, 1, nheads_kv*ngroups, d)` 转换为 `(b, ngroups, nheads_kv, d)`

#### 内存分配 (L227-278)
- **Padding**: head_size 不是 8 的倍数时，pad 到 8 的倍数
- **输出张量** `out`：复用或新建
- **softmax_lse**: `(batch, num_heads, seqlen_q)` 用于存储 logsumexp
- **dropout randval** `p`：如果需要返回 dropout mask

#### Dropout RNG 初始化 (L280-307)
- 使用 CUDA Generator 生成 Philox RNG 状态
- 调用 `ParsePhiloxCudaState` kernel 解析 seed 和 offset
- 返回 `seed_t` 和 `offset_t` 张量

#### 核心计算 (L315-352)
```cpp
auto traits = get_ck_fmha_fwd_traits(...);
auto args = get_ck_fmha_fwd_args(...);
float t = fmha_fwd(traits, args, stream_config);
```
- 调用 CK 库的 `fmha_fwd` 执行实际计算
- 特殊处理：`seqlen_k == 0` 时输出填零，lse 填充无穷大

#### 输出整理 (L359-364)
如果做了 seqlenq_ngroups_swapped 优化，需要将张量转换回原始形状

## 返回值
8 个张量的 tuple：
1. `out`: attention 输出
2. `q_padded`: padding 后的 query
3. `k_padded`: padding 后的 key
4. `v_padded`: padding 后的 value
5. `softmax_lse`: softmax logsumexp (用于反向传播)
6. `seed_t`: RNG seed
7. `offset_t`: RNG offset
8. `p`: dropout mask (可选)

---

**ROCm 相关**：
- 使用 `HIPGuardMasqueradingAsCUDA` 进行设备管理
- `hipLaunchKernelGGL` 启动 HIP kernel
- 通过 HIP stream 执行异步计算

**Backward 传播**：文件仅实现前向，反向传播需要 `softmax_lse` 和 `seed_t/offset_t` 用于重放 dropout
