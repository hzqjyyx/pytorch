这个文件实现了 PyTorch functorch 库中针对神经网络模块操作的批处理规则（batch rules），用于支持 vmap（向量化映射）功能。

## 核心功能

### 1. Embedding 批处理规则 (embedding_batch_rule)

处理嵌入层在批处理维度上的三种情况：

- **仅 indices 有批处理维度**：直接对批处理的索引进行嵌入查找
- **仅 weight 有批处理维度**：将权重的批处理维度合并到嵌入维度，查找后再分离
- **weight 和 indices 都有批处理维度**：通过 `getStepTensor` 为每个批次的索引添加偏移量（0, E, 2E, ...），将批处理权重展平为单个大嵌入表，确保每个批次访问正确的嵌入向量

### 2. Grid Sample 批处理规则 (grid_sample_batch_rule)

实现 2D/3D 网格采样的批处理，分三种情况：

- **仅 input 批处理**：将批处理维度合并到通道维度（N(BC)H_in W_in），输出在通道维度展开
- **仅 grid 批处理**：将批处理维度合并到空间维度（N(BH_out)W_out 2），输出在对应空间维度展开  
- **input 和 grid 都批处理**：将两者的批处理维度都合并到 batch 维度（(BN)CH_in W_in）

通过模板 `GridSampleBatchRuleHelper` 实现类型擦除，支持不同参数签名的 grid_sampler 函数。

### 3. Upsample 批处理规则

通过 `UpsampleBackwardBatchRuleHelper` 模板处理上采样操作，将批处理维度合并到 batch 维度，并修正 input_size 参数以匹配物理张量形状。

### 4. One-Hot 分解

`one_hot_decomposition_hack` 将 one_hot 操作分解为 `zeros + scatter`，避免在 vmap 下直接实现复杂的批处理逻辑。

### 5. 注册机制

在 `TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m)` 中注册所有批处理规则，包括：
- 卷积相关：im2col, col2im
- 空间变换：pixel_shuffle, channel_shuffle
- 填充操作：constant_pad_nd, reflection_pad, replication_pad
- 上采样：upsample_nearest, upsample_bilinear, upsample_bicubic, upsample_trilinear

**简要列出的其他内容：**
- embedding_dense_backward_batch_rule：嵌入层反向传播的批处理
- grid_sample_backward_batch_rule：网格采样反向传播的批处理
- cudnn_grid_sampler_backward：cuDNN 版本的网格采样反向传播
- upsample_*_backward：各种上采样操作的反向传播批处理规则
- ROCm 相关：ROCmFABackend.h（文件头部引用，但正文未涉及）
