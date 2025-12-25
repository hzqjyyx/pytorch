# attention.cpp/h 核心功能分析

## 主要职责

这两个文件实现了 PyTorch 中 Transformer 注意力机制的核心运算，包括多头注意力（Multi-Head Attention）和缩放点积注意力（Scaled Dot-Product Attention）。

## 关键组件

### 1. Scaled Dot-Product Attention (SDPA)

**主入口函数**: `scaled_dot_product_attention` (attention.cpp:696)

该函数实现标准的注意力计算：`Attention(Q,K,V) = softmax(QK^T/√d)V`

**后端选择机制**:
- 通过 `_fused_sdp_choice_stub` 动态选择最优后端 (attention.cpp:708-711)
- 支持多种后端实现:
  - `cudnn_attention`: cuDNN 加速实现 (attention.cpp:717)
  - `flash_attention`: Flash Attention 优化版本，CUDA 上会对头维度做 8 的倍数填充 (attention.cpp:723-737)
  - `efficient_attention`: Memory Efficient Attention (attention.cpp:739)
  - `overrideable`: 自定义后端钩子 (attention.cpp:748)
  - `math`: 朴素数学实现，作为 fallback (attention.cpp:753)

### 2. Math Fallback 实现

**函数**: `_scaled_dot_product_attention_math` (attention.cpp:814)

完整的注意力计算流程:
1. **数据类型提升**: FP16/BF16 可选提升到 FP32 以提高精度 (attention.cpp:830-847)
2. **Q/K 缩放**: 计算 `scaling_factor = 1/√d_k`，对 Q 和 K 分别缩放以提高数值稳定性 (attention.cpp:854-861)
3. **因果掩码处理**: `is_causal=True` 时生成下三角掩码 (attention.cpp:862-876)
4. **GQA/MQA 支持**: 通过 `pre_process_group_query_attention_input` 复制 K/V 的头以匹配 Q (attention.cpp:880)
5. **注意力计算**: 
   - `attn = matmul(Q, K^T) * scaling_factor` (attention.cpp:881)
   - 应用注意力掩码 (attention.cpp:882-888)
   - `_safe_softmax`: 处理全 -inf 行的特殊 softmax (attention.cpp:889)
   - Dropout (可选) (attention.cpp:890-900)
   - `output = matmul(attn, V)` (attention.cpp:903)

### 3. Native Multi-Head Attention

**函数**: `native_multi_head_attention_cpu` (attention.cpp:269)

CPU 上的完整多头注意力实现:

1. **QKV 投影**: `qkv_projection` 函数处理三种场景 (attention.cpp:193-235)
   - 自注意力: `Q=K=V`，直接用 `qkv_weight` 投影
   - 编码器-解码器注意力: `K=V≠Q`，分割权重分别投影
   - 完全不同: 分别对 Q/K/V 投影后拼接

2. **变换和缩放**: `transform_bias_rescale_qkv_cpu` (attention.cpp:238-267)
   - 将 `[B,T,3D]` 重组为 `[B,num_head,T,dim_per_head]`
   - 加偏置并缩放 Q: `q = (q + q_bias) / √dim_per_head`
   - 通过 `transform_bias_rescale_qkv_stub` 调用优化的内核

3. **注意力计算**:
   - `bmm_nt(q, k)`: 批量矩阵乘法计算 `QK^T` (attention.cpp:383)
   - `masked_softmax`: 应用掩码和 softmax (attention.cpp:395)
   - `bmm_nn(q, qkt, v)`: 计算最终输出，复用 q 的存储空间 (attention.cpp:402)

4. **输出投影**: `transform0213_gemm_nt_bias` (attention.cpp:147-163)
   - 融合维度变换 `[B,num_head,T,dim_per_head]` → `[B,T,D]`
   - 线性变换加偏置: `output = W_o * attn_output + b_o`

### 4. Flash Attention CPU 实现

**前向**: `_scaled_dot_product_flash_attention_cpu` (attention.cpp:907)
- 分配输出和 logsumexp 缓冲区
- 调用 `flash_attention_kernel` 派发到具体实现
- 转置维度以匹配输出格式

## 辅助工具函数

- `gemm_nt`: 支持 NestedTensor 的 `matmul(a, b^T)` (attention.cpp:89)
- `bmm_nt`: 4D 批量矩阵乘法 `[B,H,T,K] × [B,H,S,K]^T` (attention.cpp:108)
- `bmm_nn`: 4D 批量矩阵乘法，直接相乘 (attention.cpp:134)
- `transform_0213`: 维度置换 `[B,T,H,D] → [B,H,T,D]` 并展平 (attention.cpp:97)
- `convert_boolean_attn_mask`: 将布尔掩码转换为加性掩码，True→0.0，False→-inf (attention.cpp:525)
- `preprocess_mask`: 对 Memory Efficient Attention 填充掩码到 8 的倍数 (attention.cpp:574)
- `pad_last_dim`: Flash Attention 需要头维度是 8 的倍数的填充 (attention.cpp:595)

## 优化策略

1. **内存复用**: `bmm_nn` 复用 q 的存储空间存储输出 (attention.cpp:402)
2. **提前释放**: QKV 在使用后立即设为空以释放内存 (attention.cpp:370, 405)
3. **条件编译**: NestedTensor 特殊路径避免不必要的转换
4. **分离 Q/K 缩放**: 比直接缩放 `QK^T` 更稳定 (attention.cpp:851-861)
5. **设备特定派发**: 通过 `DECLARE_DISPATCH` 机制派发到 AVX2/AVX512/ARM 等优化内核

## 特殊特性支持

- **NestedTensor**: 动态序列长度的高效处理
- **Group Query Attention (GQA)**: K/V 头数少于 Q 时自动扩展 (attention.cpp:624)
- **因果掩码**: 自回归任务的三角掩码生成
- **混合精度**: 支持 FP16/BF16 输入，可选 FP32 累加
- **自定义掩码**: 支持布尔型和浮点型掩码

---

## 其他内容摘要

### ROCm 相关
- `_fused_sdp_choice_meta`: ROCm (HIP) 后端的设备检测 (attention.cpp:462-467)
- 条件编译 `#if defined(USE_ROCM)` 分支

### Backward 相关  
- `_scaled_dot_product_flash_attention_cpu_backward`: Flash Attention CPU 反向传播 (attention.cpp:950)
- `_scaled_dot_product_fused_attention_overrideable_backward`: 自定义后端反向钩子 (attention.cpp:1000)
- `flash_attention_backward_kernel` 派发声明 (attention.h:68)
- `should_compute_logsumexp`: 根据梯度需求决定是否计算 logsumexp (attention.cpp:618)
