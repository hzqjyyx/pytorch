# LossNLL2d.cpp 主要功能

这个文件实现了 **2D Negative Log Likelihood (NLL) Loss** 的 CPU 版本，用于处理空间数据（如图像分割任务）的损失计算。

## 核心功能

### 1. 前向传播 (Forward Pass)

**主函数**: `nll_loss2d_forward_out_frame` (102-252行)

**输入张量形状**:
- `input`: 4D 张量 `[batch_size, n_classes, H, W]` - 模型预测的 log-probabilities
- `target`: 3D 张量 `[batch_size, H, W]` - 每个像素的真实类别索引
- `weight`: 可选的 1D 张量 `[n_classes]` - 每个类别的权重

**两种计算模式**:

#### a) 无归约模式 (`Reduction::None`, 118-156行)
- 输出形状: `[batch_size, H, W]`
- 对每个像素独立计算损失
- 使用 `parallel_for` 并行处理批次
- 核心公式 (149行): `output[b][h][w] = -input[b][target[b][h][w]][h][w] * weight[target]`

#### b) 归约模式 (`Sum`/`Mean`, 158-252行)
- 输出形状: 标量
- 处理空元素特殊情况 (161-172行): 空张量 Mean 归约返回 NaN
- 使用 **级联求和** (cascade summation) 技术降低数值误差:
  - 8 层部分和数组 (185-193行)
  - 按 2 的幂次级别累积 (219-231行)
- 计算总权重并根据归约类型归一化 (246-248行)

### 2. 输入验证

**`check_inputs_nll_loss2d`** (45-75行):
- 验证 `target` 是 3D 张量
- 验证 `input` 是 4D 张量
- 验证权重维度匹配类别数
- 验证 batch/spatial 维度匹配: `input[0,2,3] == target[0,1,2]`

### 3. 特殊处理

- **`ignore_index`**: 忽略特定类别索引，不计入损失 (134-136行, 199-201行)
- **边界检查**: 验证 target 值在 `[0, n_classes)` 范围内 (139-143行, 204-208行)
- **可选权重**: 通过 `optional_contiguous` 和 `optional_data` 辅助函数处理 (30-43行)

### 4. 类型支持

使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持多种数据类型:
- 浮点类型: `float`, `double`
- 半精度类型: `BFloat16`, `Half`

### 5. 公共 API

- `nll_loss2d_forward_out_cpu` (407-420行): 带输出参数版本
- `nll_loss2d_forward_cpu` (422-432行): 返回新张量版本
- `nll_loss2d_out` (477-480行): 简化接口
- `nll_loss2d_symint` (482-484行): 支持符号整数的版本

## 关键优化

1. **并行化**: 使用 `at::parallel_for` 对批次维度并行处理
2. **数值稳定性**: 级联求和减少浮点累积误差
3. **内存连续性**: 通过 `contiguous()` 确保数据局部性
4. **条件分支优化**: `C10_LIKELY` 提示编译器优化热路径 (222行)

---

**ROCm 相关**: 无（纯 CPU 实现）

**Backward 相关**:
- `nll_loss2d_backward_out_frame` (282-364行) - 反向传播核心实现
- `nll_loss2d_backward_out_cpu` (434-455行) - 带输出参数版本
- `nll_loss2d_backward_cpu` (457-475行) - 返回新张量版本
- 梯度计算公式: `grad_input[b][t][h][w] = -weight[t] * grad_output[b][h][w] / total_weight` (Mean 模式)
