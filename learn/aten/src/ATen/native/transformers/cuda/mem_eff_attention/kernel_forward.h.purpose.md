# kernel_forward.h 主要功能分析

这是 PyTorch 中 Memory-Efficient Attention 的前向传播 CUDA 核心实现，基于 CUTLASS 库实现了高效的注意力机制计算。

## 核心架构

### AttentionKernel 模板类 (90-1338行)
主要模板参数：
- `scalar_t_`: 数据类型 (fp16/bf16/fp32)
- `ArchTag`: 目标架构 (Sm70/Sm75/Sm80等)
- `isAligned_`: 内存对齐标志，影响性能优化路径
- `kQueriesPerBlock_`, `kKeysPerBlock_`: Tile 大小，决定每个 threadblock 处理的数据块
- `kMaxK_`: head_dim 上界
- `kSupportsDropout_`, `kSupportsBias_`: 功能开关

### 计算流程 (attention_kernel 函数，624-1179行)

**1. 初始化阶段 (629-647行)**
- 分配共享内存 (`SharedStorage`)
- 初始化 softmax 相关变量：`m_prime`, `s_prime`, `mi`, `out_rescale`
- 初始化输出累加器 `accum_o`

**2. 主循环：按 Key blocks 迭代 (700-1117行)**

每次迭代处理 `kKeysPerBlock` 个 keys：

**a) MM0: Q @ K^T 矩阵乘法 (734-784行)**
- 使用 `MM0::Mma` 执行 Q 和 K 的转置矩阵乘
- 结果存储在寄存器 `accum` 中
- 关键配置：
  - ThreadblockShape: `[kQueriesPerBlock, kKeysPerBlock, ThreadK]`
  - WarpShape: `[32, 32, WarpK]`

**b) Scaling & Bias (799-833行)**
- 乘以缩放因子 `scale`
- 如果启用，加上注意力偏置 `attn_bias`
  - 通过 `BiasLoader` 从全局内存加载到共享内存
  - 然后加到 `accum` 寄存器中

**c) Masking (835-898行)**
- **Causal masking** (842-865行)：
  - 根据 `custom_mask_type` 应用因果掩码
  - 将超出对角线的位置设为 `-inf`
- **Sliding window** (874-898行)：
  - 如果 `window_size > 0`，掩盖窗口外的注意力

**d) Iterative Softmax (902-916行)**
调用 `iterative_softmax` 函数 (1182-1327行) 实现在线 softmax：
- **第一遍**：更新每行最大值 `mi` (1226-1243行)
  - 使用 `atomicMaxFloat` 跨线程更新
- **第二遍** (1248-1327行)：
  - 计算 `exp(accum - mi)`，防止数值溢出
  - 更新缩放系数 `out_rescale = exp(m_prime - mi)`
  - 累加 softmax 分母 `s_prime`
  - 如果保持输出在寄存器中，同步更新 `frag_o`

**e) Dropout (930-989行)**
如果启用：
- 使用 Philox RNG 生成随机数
- 每个线程处理连续的同一行元素（提高缓存效率）
- 以 `dropout_scale = 1/(1-p)` 缩放，或置零

**f) MM1: Attn @ V 矩阵乘法 (991-1115行)**
- 从共享内存读取注意力分数 `si`
- 从全局内存读取 Value `V`
- 使用 `MM1::Mma` 执行矩阵乘
- 结果累加到 `accum_o`

**g) Epilogue (1038-1114行)**
如果不保持输出在寄存器中 (`!kKeepOutputInRF`)：
- 使用 `EpiloguePipelined` 进行归一化
- 根据是否为首次/末次迭代选择不同的输出类型
- 应用 `out_rescale` 进行最终缩放
- 写入全局内存

**3. 最终输出 (1119-1158行)**
如果 `kKeepOutputInRF` (单次 Value 迭代)：
- 在所有迭代完成后，一次性执行 epilogue
- 归一化并写入最终输出

**4. 计算 LogSumExp (1160-1178行)**
如果需要 (反向传播用)：
- 计算 `lse = mi/log2(e) + log(s_prime)`
- 处理完全掩盖的行 (设为0避免 NaN)
- 填充到对齐边界，用 `inf` 填充

## 内存优化策略

### SharedStorage 布局 (514-570行)
使用 union 复用共享内存：
- `mm0`: MM0 的共享内存
- `after_mm0`: MM0 后的内存，包含
  - `bias` / `si`: 复用同一块内存
  - `mm1`: MM1 的共享内存
  - `epilogue`: epilogue 的共享内存（根据配置决定位置）

### 两种存储策略
- `SharedStorageEpilogueAtEnd`: epilogue 与 mm0 共享（单次迭代）
- `SharedStorageEpilogueInLoop`: epilogue 在循环内（多次迭代）

### 寄存器 vs 共享内存
- `kKeepOutputInRF = kSingleValueIteration`: 单次 V 迭代时，输出保持在寄存器中，避免读写共享内存
- `kPreloadV`: 在 Sm80+ 和半精度时启用，预加载 Value 到共享内存

## Params 结构 (132-345行)

### 核心字段
- 输入张量指针：`query_ptr`, `key_ptr`, `value_ptr`, `attn_bias_ptr`
- 输出指针：`output_ptr`, `output_accum_ptr`, `logsumexp_ptr`
- 维度/步长：`head_dim`, `num_queries`, `num_keys`, 各种 stride
- 配置：`scale`, `window_size`, `custom_mask_type`, dropout 参数

### advance_to_block (198-333行)
关键的指针推进逻辑：
- 根据 `batch_id`, `head_id`, `query_start` 调整指针
- 处理变长序列 (通过 `seqstart_q/k_ptr`, `seqlen_k_ptr`)
- 应用因果掩码的偏移
- **特殊优化** (300-313行)：当 `num_queries==1` 且 key/value 是单头时，沿 head 维度迭代而非 query 维度

### check_supported (572-622行)
验证参数对齐要求：
- Query 对齐到 `kAlignmentQ`
- Key 对齐到 `kAlignmentK`
- Value 对齐到 `kAlignmentV`
- 各种 stride 对齐检查

## 数值稳定性

### Online Softmax 算法
采用 "online normalization" 避免两遍遍历：
- 维护运行最大值 `mi` 和运行和 `s_prime`
- 公式：`s_prime_new = s_prime * exp(m_prime - mi_new) + sum(exp(x - mi_new))`
- 避免 `exp` 溢出，保证数值稳定

### 特殊情况处理
- 全掩盖行：`s_prime==0` 时设为1，`mi==-inf` 时设为0 (1169-1171行)
- Bias 导致的 `-inf - (-inf)`：临时设 `mi=0` 避免 NaN (1264-1268行)

## 并行策略

### 网格配置
- `blockIdx.x`: query block 索引
- `blockIdx.y`: head 索引
- `blockIdx.z`: batch 索引
- 每个 threadblock: `kWarpSize * kNumWarpsPerBlock` 个线程

### Warp 级别并行
- MM0/MM1 使用 warp-level GEMM
- Softmax reduction 使用 warp 内规约
- 不同 warp 处理不同的 output tile

---

## 简要提及的其他内容

**ROCm 相关**：
- 未在此文件中出现（主要在其他 ROCm-specific 实现中）

**Backward 相关**：
- `logsumexp_ptr` 用于保存 log-sum-exp，供反向传播使用
- `extragraph_offset`, `seed` 用于 CUDA Graph 模式下的 RNG 状态保存
