这个文件实现了 PyTorch functorch 库中归一化操作（normalization）的批处理规则（batch rules），用于支持 vmap（向量化映射）功能。

## 核心功能

### 1. Batch Normalization 的批处理规则

**前向传播** (`batch_norm_batch_rule`, 44-125行):
- 处理输入张量带有批次维度（batch dimension）的情况
- 关键策略：将批次维度重塑到通道维度中
  - 例如：`[B0, C, H, W]` → `[(B0*C), H, W]`
  - 对重塑后的张量调用原始 batch norm
  - 将结果重塑回 `[B0, C, H, W]`
- 处理 `running_mean` 和 `running_var` 的批次维度
- 训练模式下检查：如果输入有批次维度，running stats 也必须有批次维度
- 分别处理 weight 和 bias 的批次维度，使用 `padRight` 进行形状对齐

**统计量处理** (`compute_stat_bdim`, 19-29行):
- 计算 mean 和 rstd（reciprocal standard deviation）的批次维度
- 处理空张量的特殊情况

### 2. Group Normalization 的批处理规则

**前向传播** (`native_group_norm_plumbing`, 292-342行):
- 如果输入有批次维度，将其重塑到 batch 维度：`N * bdim_size`
- 调用原始 group norm，然后重塑回原始形状
- 应用 weight 和 bias（如果定义）

### 3. Layer Normalization 的批处理规则

**前向传播** (`native_layer_norm_batch_rule`, 490-530行):
- 将批次维度移到最前面
- 如果 weight/bias 没有批次维度，直接调用原始 layer norm
- 如果 weight/bias 有批次维度：
  - 先不带 weight/bias 计算归一化
  - 然后手动应用带批次维度的 weight 和 bias
  - 使用 `maybePadToLogicalRank` 进行形状对齐

**输入验证** (`_check_layer_norm_inputs`, 475-488行):
- 检查 normalized_shape 至少是 1 维
- 验证 weight 和 bias 的形状与 normalized_shape 匹配

### 4. 辅助函数

**`padRight`** (31-42行):
- 在张量右侧填充维度以匹配目标逻辑秩
- 用于对齐 weight/bias 与输入张量的形状

**`has_same_shape` / `check_same_shape`** (440-472行):
- 验证张量形状（考虑批次维度）是否与 normalized_shape 匹配

### 5. 后端特化

支持三种后端的 batch norm 实现：
- **Native**: 标准 CPU/CUDA 实现
- **cuDNN**: NVIDIA 优化实现（额外返回 reserve 张量）
- **MIOpen**: AMD ROCm 优化实现

通过模板和宏定义统一处理：
- `NativeBatchNormBatchRuleHelper`
- `CudnnBatchNormBatchRuleHelper`
- `MiopenBatchNormBatchRuleHelper`

### 6. 注册机制

在文件末尾（867-880行）通过 `TORCH_LIBRARY_IMPL` 注册所有批处理规则到 `FuncTorchBatched` 调度键。

---

**ROCm 相关内容**:
- `MiopenBatchNormBatchRuleHelper`: MIOpen batch norm 的批处理规则辅助类
- `MIOPEN_BATCH_NORM_BATCH_RULE` / `MIOPEN_BATCH_NORM_BACKWARD_BATCH_RULE`: 宏定义
- `miopen_batch_norm_backward_wrapper`: MIOpen 反向传播包装函数

**Backward 相关内容**:
- `batch_norm_backward_no_weight_bias_batch_rule` (129-198行): batch norm 反向传播（不计算 weight/bias 梯度）
- `batch_norm_backward_plumbing` (202-290行): batch norm 反向传播完整实现，计算 grad_input, grad_weight, grad_bias
- `group_norm_backward_no_weight_bias_batch_rule` (344-373行): group norm 反向传播
- `native_group_norm_backward_plumbing` (375-438行): group norm 反向传播完整实现
- `native_layer_norm_backward_no_weight_bias_batch_rule` (532-567行): layer norm 反向传播
- `native_layer_norm_backward_plumbing` (569-650行): layer norm 反向传播完整实现
- 各种 `*BackwardBatchRuleHelper` 模板类用于统一处理不同后端的反向传播
