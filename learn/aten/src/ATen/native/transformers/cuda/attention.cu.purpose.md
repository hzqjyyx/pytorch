## 文件主要功能分析

### 核心功能模块

#### 1. **QKV 变换与预处理**
提供两个 CUDA kernel 实现 QKV 张量的变换：
- `transform_bias_rescale_qkv_kernel`: 处理标准张量
  - 计算 `q = (q + q_bias) / sqrt(dim_per_head)` (缩放用于数值稳定)
  - 计算 `k = k + k_bias`, `v = v + v_bias`
  - 形状变换: [B, T, 3×D] → [3, B, NH, T, DH]
  - 支持向量化内存访问 (VEC=4)，提升吞吐

- `transform_bias_rescale_qkv_add_padding_kernel`: NestedTensor 版本
  - 处理可变长度序列，自动填充 padding
  - 序列长度对齐到 8 的倍数以利用 Tensor Core
  - 边界外填充 0，避免影响计算

#### 2. **原生多头注意力实现**
`native_multi_head_attention_cuda` - 完整的标准 MHA 流程:

```
Input → Linear(qkv_weight, qkv_bias) → Transform 
  → Q·K^T → Masked Softmax → Attention·V 
  → Linear(proj_weight, proj_bias) → Output
```

**快速路径优化**: 当 query/key/value 是同一张量且 `dim_per_head % 8 == 0` 时，直接调用融合 SDP 后端（Flash/Efficient/cuDNN Attention），跳过传统计算路径。

#### 3. **三种高性能 Attention 后端**

**Flash Attention** (`_flash_attention_forward`)
- 使用第三方 Flash Attention 库
- 两种模式：标准 MHA (`mha_fwd`) 和可变长序列 (`mha_varlen_fwd`)
- 支持窗口注意力、ALiBi 位置编码
- 返回 logsumexp 和随机数状态（用于 backward 重计算 dropout）

**cuDNN Attention** (`_cudnn_attention_forward`)
- 调用 NVIDIA cuDNN 的 SDPA API
- 支持标准张量和 NestedTensor
- Attention bias 自动扩展 (2D/3D/4D → 4D)
- CUDA Graph 兼容的 Philox 随机数生成器

**Memory Efficient Attention** (`_efficient_attention_forward`)
- CUDA 版本使用 CUTLASS 库实现
- **动态 kernel 分发**: 通过 `launchKernel` lambda 根据条件选择最优 kernel:
  - 检查 dropout/bias 支持
  - 检查 head dimension 上限 (`kMaxK`)
  - 检查内存对齐要求 (`kAlignmentQ/K/V`)
  - 检查 shared memory 使用量
- 支持可变长序列 (`seqstart_q/k`) 和窗口注意力

#### 4. **智能后端选择**
`_fused_sdp_choice_cuda`: 根据硬件能力和输入特征自动选择最优后端（Flash/Efficient/cuDNN/Math）

#### 5. **辅助功能**
- `unpack_cudnn`: CUDA Graph 安全的随机数状态解包
- `collapse_dims_1_and_2`: NestedTensor 维度折叠工具
- Philox RNG 状态管理: 支持动态 graph capture

### 关键优化技术

1. **向量化内存访问**: `aligned_vector<4>` 提升带宽利用率
2. **Tensor Core 对齐**: 序列长度 padding 到 8 的倍数
3. **kernel 动态分发**: 根据数据类型、对齐、硬件特征选择最优实现
4. **操作融合**: bias + rescale + reshape 融合在单个 kernel
5. **NestedTensor 优化**: 避免固定长度 padding 的计算浪费
6. **CUDA Graph 兼容**: 随机数生成器使用设备端种子/偏移量指针

---

### ROCm 相关内容
- 头文件引入: `#ifdef __HIP_PLATFORM_AMD__` 分支使用 HIP 版本的 MHA 头文件
- Memory Efficient Attention 的 ROCm 实现路径 (1245-1386 行):
  - CK (Composable Kernel) 后端
  - AOTriton 后端（支持 MI200/MI300X/Navi31 GPU）
- Dropout mask 填充的 ROCm 专用实现
- 使用 `hipError_t` 替代 `cudaError_t`

### Backward 相关内容
- 随机数种子/偏移量的保存用于 backward 重建 dropout mask
- Logsumexp 的内存对齐考虑了 backward 加载效率
- 实际的 backward kernel 实现在其他文件中
