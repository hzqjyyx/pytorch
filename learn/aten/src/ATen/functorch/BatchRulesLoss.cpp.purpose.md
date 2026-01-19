这个文件实现了 PyTorch functorch 库中损失函数的批处理规则（batch rules），用于支持 vmap（向量化映射）操作。

## 核心机制

**flatten_logical 函数** (15-26行)
- 将张量展平，但保留批次维度
- 如果存在批次维度，先移到最前面，然后展平其余维度
- 如果没有批次维度，直接全部展平

**loss_batch_rule_helper 模板函数** (29-50行)
- 通用的损失函数批处理规则辅助函数
- 处理流程：
  1. 对 self 和 target 张量进行逻辑展平
  2. 调用底层损失函数，强制使用 `Reduction::None`
  3. 根据原始的 reduction 参数处理结果：
     - `Reduction::None`: 重塑回原始批次形状
     - `Reduction::Sum`: 对最后一维求和
     - `Reduction::Mean`: 对最后一维求均值
  4. 返回结果张量和批次维度索引 0

## 具体损失函数实现

**mse_loss_batch_rule** (52-59行)
- MSE（均方误差）损失的批处理规则
- 直接使用 helper 函数包装 `at::mse_loss`

**huber_loss_batch_rule** (61-68行)
- Huber 损失的批处理规则
- 通过 lambda 捕获 delta 参数

**smooth_l1_loss_batch_rule** (70-77行)
- Smooth L1 损失的批处理规则
- 通过 lambda 捕获 beta 参数

**binary_cross_entropy_plumbing** (88-123行)
- 二元交叉熵的特殊处理函数
- 手动处理批次维度对齐和权重应用
- 支持可选的 weight 参数

**apply_loss_reduction** (79-86行)
- 辅助函数，应用最终的 reduction 操作

## 注册机制

**TORCH_LIBRARY_IMPL** (174-183行)
- 将批处理规则注册到 FuncTorchBatched 调度键
- 使用 `VMAP_SUPPORT` 宏注册标准损失函数
- 使用 `m.impl` 直接注册需要特殊处理的函数

---

**Backward 相关内容：**
- `binary_cross_entropy_backward_plumbing` (125-172行)：处理二元交叉熵反向传播的批处理规则

**ROCm 相关内容：**
- 文件中未涉及 ROCm 特定代码
