# Flash Attention 实现分析

## 核心功能

这是 PyTorch 在 AMD GPU (ROCm) 上实现 Flash Attention 的适配层，通过 AOTriton 库提供高效的 Multi-Head Attention (MHA) 计算。

## 主要函数

### 1. `mha_fwd_aot` - 标准前向计算
**位置**: aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip:122

**功能**: 对固定形状的 batch 进行 attention 前向计算

**核心流程**:
```
输入验证 → 张量重排列 → 调用 AOTriton API → 返回结果
```

**关键操作**:
- 检查数据类型 (仅支持 fp16/bf16)
- 检查 head_size 必须是 8 的倍数，最大 512
- 检查 GPU 架构 (MI200/MI300X/Navi31)
- 将 PyTorch 张量 permute 为 AOTriton 所需的 `(batch, heads, seqlen, head_dim)` 格式
- 处理 dropout 的随机数生成 (Philox RNG)
- 调用 `aotriton::v2::flash::attn_fwd`

**返回值**:
```cpp
{out, q_padded, k_padded, v_padded, M, seed_t, offset_t, softmax_fa_t}
```
- `M`: softmax logsumexp，用于后向传播
- `seed_t`, `offset_t`: dropout 的 RNG 状态

### 2. `mha_varlen_fwd_aot` - 变长序列前向计算
**位置**: aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip:251

**功能**: 处理不同长度序列的 batch (通过 `cu_seqlens` 索引)

**与标准版本的区别**:
- 输入形状: `(total_tokens, num_heads, head_dim)` 而非 4D
- 需要 `cu_seqlens_q/k`: 累积序列长度数组 (batch_size+1)
- 使用 `unsqueeze(0).transpose(1,2)` 将张量调整为 AOTriton varlen API 所需格式
- 调用 `aotriton::v2::flash::attn_fwd_compact_varlen`
- 支持空序列处理 (`max_seqlen_k == 0`)

**典型应用场景**: 
LLM 推理中的动态 batching，不同样本有不同的序列长度

## 辅助函数

### `prepare_philox_arguments`
**位置**: aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip:87

**功能**: 准备 dropout 的随机数生成器状态

**关键点**:
- 从 CUDA Generator 获取 Philox 状态
- 区分普通执行 vs CUDA Graph 捕获模式
- CUDA Graph 模式下使用指针而非立即值 (见代码注释 "CUDA Graph-safe RNG states")

### `check_gpu_arch`
**位置**: aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip:75

**功能**: 验证 GPU 架构是否支持

**支持的架构**:
- gfx90a (MI200 系列)
- gfx942 (MI300X)
- gfx1100 (Navi31)

## AOTriton 适配细节

### 张量包装
使用 `sdp::aotriton_adapter` 命名空间的工具函数:
- `mk_aotensor`: 包装普通张量
- `mk_aoscalartensor`: 包装标量张量
- `mk_philoxtensor`: 包装 Philox RNG 指针
- `mk_atomictensor`: 包装原子计数器 (用于 causal attention)

### 空 bias 处理
```cpp
aotriton::TensorView<4> empty_bias(0, {0,0,0,0}, {0,0,0,0}, cast_dtype(q.dtype()));
```
Flash Attention 不使用 attention bias，传递空视图

## 编译条件

```cpp
#ifdef USE_FLASH_ATTENTION
#if AOTRITON_VERSION_MINOR != 9
#error "This adaptor code is only tested with AOTriton 0.9.x"
#endif
```

仅在编译时启用 Flash Attention 且 AOTriton 版本为 0.9.x 时生效

---

## ROCm 相关特性 (简要)
- 使用 `HIPGuardMasqueradingAsCUDA` 设备管理
- 使用 `getCurrentHIPStreamMasqueradingAsCUDA()` 获取流
- 通过 hipError_t 处理 HIP API 调用

## Backward 函数 (简要)
- `mha_bwd_aot`: 标准后向传播，支持 fused/unfused 两种模式
- `mha_varlen_bwd_aot`: 变长序列后向传播
- 根据 `d_head * seqlen_q < 64 * 512` 选择 fused kernel
- 需要前向传播保存的 `softmax_lse` 和 RNG 状态
