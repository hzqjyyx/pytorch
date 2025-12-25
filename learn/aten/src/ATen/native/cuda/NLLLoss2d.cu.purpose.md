# NLLLoss2d.cu 核心功能分析

这个文件实现了 **2D Negative Log Likelihood Loss (NLL Loss)** 的 CUDA 加速计算，专门用于处理空间标签（spatial targets）的场景，如图像分割任务。

## 输入输出格式

- **输入张量**: 4D `(N, C, H, W)` - 批次大小、类别数、高度、宽度
- **目标张量**: 3D `(N, H, W)` - 每个空间位置的类别标签
- **权重张量**: 可选的 1D `(C,)` - 每个类别的权重

## 核心计算逻辑

对于输入 `input[b][c][h][w]` 和目标 `target[b][h][w] = t`，损失计算为:
```
loss[b][h][w] = -input[b][t][h][w] * weight[t]
```

即从每个空间位置的类别预测中，根据真实标签 `t` 选择对应通道的值，取负并乘以权重。

## 三种归约模式

### 1. None (不归约)
- 内核: `nll_loss2d_forward_no_reduce_kernel` (aten/src/ATen/native/cuda/NLLLoss2d.cu:50-78)
- 输出: 3D `(N, H, W)`，保留每个位置的损失值
- 并行策略: 按 `(batch, h, w)` 扁平化索引并行

### 2. Sum (求和)
- 内核: `nll_loss2d_forward_kernel` (aten/src/ATen/native/cuda/NLLLoss2d.cu:82-125)
- 输出: 标量，所有有效损失的总和
- 并行策略:
  - 每个样本分配多个 block (`blocks_per_sample`)
  - Block 内使用共享内存归约 (`BlockReduceSum`)
  - 原子加操作汇总到全局输出

### 3. Mean (平均)
- 先执行 Sum 模式
- 再调用 `nll_loss2d_forward_size_average_kernel` (aten/src/ATen/native/cuda/NLLLoss2d.cu:129-134)
- 除以总权重: `output /= total_weight`

## 关键技术细节

**ignore_index 处理**:
- 当 `target[b][h][w] == ignore_index` 时，该位置损失为 0 且不计入权重累加
- 用于忽略填充位置或未标注区域

**原子操作与非确定性**:
- Sum/Mean 模式使用 `gpuAtomicAdd` 汇总结果 (aten/src/ATen/native/cuda/NLLLoss2d.cu:122-123)
- 会触发非确定性警告 (aten/src/ATen/native/cuda/NLLLoss2d.cu:242)

**索引类型优化**:
- 根据张量大小动态选择 `int` 或 `int64_t` 索引 (aten/src/ATen/native/cuda/NLLLoss2d.cu:325)
- 小张量用 32 位索引提升性能

**边界情况处理**:
- 空张量 `numel() == 0`: Mean 返回 NaN，Sum 返回 0 (aten/src/ATen/native/cuda/NLLLoss2d.cu:291-301)
- 自动跳过空张量的内核启动

## 补充信息

**ROCm 相关**: 无 ROCm 特定代码，通用 CUDA 实现

**Backward 功能**:
- `nll_loss2d_backward_no_reduce_kernel`: 计算 `grad_input[b][t][h][w] = -weight[t] * grad_output[b][h][w]`
- `nll_loss2d_backward_kernel`: 归约模式下额外除以 `total_weight`
- 反向传播只更新目标类别对应的通道，其他通道梯度为 0
