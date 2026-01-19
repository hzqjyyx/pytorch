这个文件实现了 PyTorch 的 **vmap (vectorized map) 功能的批处理规则 (batching rules)**。

## 核心概念

**Batching Rule** 定义了当张量带有额外的批次维度时，如何调用算子。vmap 会将被映射的维度记录为批次维度，batching rule 负责将逻辑张量转换为物理张量，执行操作后再转换回来。

## 工作流程

典型的 batching rule 遵循四步模式：

1. **逻辑到物理转换**：使用 `MultiBatchVmapTransform::logicalToPhysical()` 将 BatchedTensor 转换为物理视图
2. **参数转换**：使用 `getPhysicalDim()` / `getPhysicalShape()` 将逻辑维度/形状转换为物理维度/形状
3. **执行操作**：在物理张量上调用 `at::` 操作
4. **结果转换**：使用 `getPhysicalToLogicalMap().apply()` 将物理结果转换回 BatchedTensor

## 主要实现的算子类别

**视图操作**：
- `expand`, `reshape`, `view`, `transpose`, `permute`, `select`, `slice`
- `squeeze`, `unsqueeze`, `chunk`, `split`, `unbind`
- `diagonal`, `unfold`, `movedim`
- `as_strided`（复杂实现，包含内存安全检查）

**逐点操作**：
- 一元：`abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `sigmoid`, `tanh` 等
- 二元：`add`, `sub`, `mul`, `div`, `pow`
- 比较：`eq`, `gt`, `ge`, `le`, `lt`, `ne`

**矩阵操作**：
- `mv`, `mm`, `bmm`, `dot`
- 特殊处理：避免不必要的维度扩展以提升性能

**归约操作**：
- `sum`（处理标量张量的特殊情况）
- `trace`

**拼接操作**：
- `cat`, `stack`

**原地操作**：
- `fill_`, `zero_`

**其他**：
- `clone`, `contiguous`
- `new_zeros`, `new_empty`, `new_empty_strided`
- `to` 系列类型转换

## 特殊处理

**标量张量**：PyTorch 允许在标量张量上使用 `dim=0` 或 `dim=-1`，batching rule 需要复制这种行为

**类型提升**：二元逐点操作需要模拟 TensorIterator 的类型提升行为，特别是当一个操作数是逻辑标量时

**内存布局**：`as_strided` 实现包含复杂的安全检查，确保批次维度在内存布局前端，且不会访问越界内存

**广播**：使用 `BroadcastingVmapTransform` 处理需要广播的操作

## 注册机制

通过 `TORCH_LIBRARY_IMPL(aten, Batched, m)` 将所有 batching rules 注册到 Batched dispatch key，当张量带有批次维度时自动调用相应规则。

---

**简要列出的其他内容**：
- Backward 算子：`select_backward`, `slice_backward`, `trace_backward`, `diagonal_backward`, `sigmoid_backward`, `threshold_backward`
- ROCm 相关：文件中未涉及 ROCm 特定代码
