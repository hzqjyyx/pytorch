# LossNLL.cpp 核心功能分析

这个文件实现了 **NLL (Negative Log Likelihood) Loss** 的计算逻辑，是 PyTorch 中分类任务损失函数的核心实现。

## 主要实现内容

### 1. **元信息定义 (Meta Functions)**
- `nll_loss_forward` (lines 37-81): 定义前向传播的输出形状和维度验证
  - 检查输入必须是 1D 或 2D 张量
  - 检查目标必须是 0D 或 1D（不支持多目标）
  - 验证 batch size 匹配
  - 验证 weight 张量维度正确
  - 根据 reduction 类型设置输出形状（标量或向量）

### 2. **前向传播核心实现**
`nll_loss_out_frame<scalar_t, target_t>` (lines 157-297) 是计算核心：

**处理 `Reduction::None` 且输入为 2D 的情况** (lines 175-204):
- 为每个样本单独计算损失，输出形状为 `[batch_size]`
- 使用并行循环遍历 batch
- 对每个样本：`loss[i] = -input[i][target[i]] * weight[target[i]]`
- 忽略 `ignore_index` 对应的样本（loss 设为 0）

**处理 reduction 为 Sum/Mean 的情况** (lines 207-297):
- 输出标量
- **空张量特殊处理** (lines 210-221): Mean reduction 返回 NaN，Sum 返回 0
- **级联求和优化** (lines 232-278): 使用 8 层级联部分和来提高数值稳定性
  - 避免大量浮点数直接累加导致的精度损失
  - 类似于树形归约的思想
- 计算公式：
  - `loss = -Σ(input[i][target[i]] * weight[target[i]])`
  - Mean reduction: `loss /= total_weight`

### 3. **Cross Entropy Loss 实现**
Cross Entropy = Log Softmax + NLL Loss

**概率目标版本** `cross_entropy_loss_prob_target` (lines 490-556):
- 输入和目标形状相同时使用（软标签/概率分布作为目标）
- 计算 `loss = -Σ(log_softmax(input) * target * weight)`
- 支持 label smoothing: `target' = target * (1-ε) + ε/n_classes`

**标签平滑版本** `cross_entropy_loss_label_smoothing` (lines 558-622):
- 整数标签 + label smoothing
- 组合 NLL loss 和平滑损失：
  - `loss = (1-ε) * nll_loss + ε/n_classes * smooth_loss`
  - `smooth_loss = -Σ(log_softmax(input))`，忽略 ignore_index 位置

**统一入口** `cross_entropy_loss_symint` (lines 624-659):
- 自动检测目标类型（概率分布 vs 整数标签）
- 路由到对应实现

### 4. **高维支持**
`nll_loss_nd_symint` (lines 670-734):
- 处理任意维度输入（1D, 2D, 3D, 4D+）
- 3D/5D+ 输入会被重塑为 4D 后调用 `nll_loss2d`
- 维度映射：`[N, C, d1, d2, ...] -> [N, C, 1, d1*d2*...]`

## 关键技术细节

### 数值稳定性优化
- **级联部分和** (lines 232-278): 8 层级联求和减少浮点累加误差
- **空张量处理**: Mean reduction 正确返回 NaN（符合数学定义）

### 性能优化
- 使用 `at::parallel_for` 并行处理 batch 维度
- 连续内存访问：提前调用 `contiguous()`
- 使用 `accessor` API 减少边界检查开销

### 类型支持
- 浮点类型：Float, Double, Half, BFloat16
- 目标类型：int64_t, uint8_t (byte)

---

## 简要说明（忽略的部分）

**Backward 相关**:
- `nll_loss_backward_out_frame` (lines 336-409): 计算梯度，公式为 `grad_input[i][t] = -weight[t] * grad_output[i]`
- `nll_loss_backward_out_cpu_template` (lines 411-451): 反向传播模板函数

**ROCm 相关**: 
- 无 ROCm 特定代码，仅 CPU 实现
