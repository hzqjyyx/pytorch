# LegacyVmapTransforms 主要功能

这两个文件实现了 PyTorch functorch 中 vmap（向量化映射）的核心转换机制，负责在**逻辑视图**和**物理视图**之间转换张量。

## 核心概念

### 逻辑视图 vs 物理视图

以 `vmap(vmap(func, in_dims=(2,)), in_dims=(0,))(torch.ones(2, 3, 4))` 为例：

- **物理张量**：实际存储的 `ones(2, 3, 4)`，批次维度分散在位置 0 和 2
- **逻辑张量**：func 内部看到的形状为 `[3]`，批次维度被隐藏
- **BatchedTensor 表示**：`BatchedTensor(ones(2, 3, 4), bdims=[(lvl=1,dim=0),(lvl=2,dim=2)])`

类似地，维度索引也需要转换：用户在 func 内调用 `torch.sum(tensor, dim=0)` 时，逻辑维度 0 对应物理维度 1（第一个非批次维度）。

## 主要组件

### 1. VmapPhysicalView

**作用**：表示张量的物理视图，所有批次维度移到最前面

**核心数据**：
- `tensor_`：批次维度已前置的物理张量
- `levels_`：bitset，标记哪些 vmap 层级对应前面的批次维度

**示例**：
```cpp
VmapPhysicalView(tensor=ones(2, 3, 4, 5, 6), levels={1, 3})
// levels bitset: 010100
//                   ^
// 表示前 2 个维度是批次维度，对应 vmap level 1 和 3
```

**主要方法**：
- `getPhysicalDims(logical_dims)`：逻辑维度 → 物理维度（加上批次维度数量的偏移）
- `getPhysicalShape(logical_shape)`：逻辑形状 → 物理形状（前面加上批次大小）
- `numBatchDims()`：返回批次维度数量（levels bitset 中设置的位数）

### 2. MultiBatchVmapTransform

**用途**：处理具有多个批次维度的算子

**转换流程**：

对单个张量：
```cpp
logicalToPhysical(logical_tensor) {
  1. 获取 BatchedTensorImpl
  2. 调用 permuteBatchDimsToFront() 将批次维度移到最前面
  3. 返回 VmapPhysicalView
}
```

对张量列表（更复杂）：
```cpp
logicalToPhysical(logical_tensors) {
  1. 确定当前 vmap level 和批次大小 bdim_size
  2. 对每个张量：
     - 如果有批次维度：movedim 到位置 0
     - 如果没有批次维度：unsqueeze(0) 然后 expand 到 bdim_size
  3. 确保所有张量的批次维度对齐并具有相同大小
}
```

**关键点**：通过 expand 操作使非批次张量与批次张量对齐

### 3. BroadcastingVmapTransform

**用途**：处理需要广播的算子（如逐元素运算）

**与 MultiBatchVmapTransform 的区别**：
- MultiBatch：用 `expand` 扩展批次大小
- Broadcasting：用 `unsqueeze` 添加大小为 1 的维度，让后续的广播操作自动处理

**转换流程**：
```cpp
logicalToPhysical(logical_tensors) {
  1. 确定批次大小和最大逻辑维度数 max_example_dim
  2. 对每个张量调用 moveDimToFrontAndUnsqueeze：
     - 批次维度移到位置 0（或 unsqueeze(0)）
     - 在位置 1 插入 unsqueeze，直到非批次维度对齐
}
```

**示例**：
- 输入：`(B, 2)` 和 `(B, 3, 2)` → 输出：`(B, 1, 2)` 和 `(B, 3, 2)`
- 输入：`(B, 2)` 和 `(2,)` → 输出：`(B, 2)` 和 `(1, 2)`

### 4. VmapPhysicalToLogicalMap

**作用**：物理张量 → 逻辑张量（BatchedTensor）的反向转换

**核心方法**：
```cpp
apply(physical_tensor) {
  1. 从 levels_ bitset 计算 bdim 和 level
  2. 调用 makeBatched(physical_tensor, bdim, level)
  3. 返回包装后的 BatchedTensor
}
```

## 辅助函数

- `permuteBatchDimsToFront()`：对 BatchedTensorImpl 进行维度置换，将批次维度移到最前面
- `moveDimToFrontAndExpand()`：移动维度到位置 0，或 unsqueeze + expand
- `moveDimToFrontAndUnsqueeze()`：移动维度到位置 0，然后添加 unsqueeze 对齐
- `computeFrontBatchDimsFromLevels()`：从 levels bitset 计算第一个批次维度的位置和层级

## 工作流程总结

1. **批处理规则调用转换**：将逻辑参数转为物理参数
2. **调用底层 ATen 算子**：使用物理张量执行实际计算
3. **反向转换**：将物理结果转回逻辑 BatchedTensor

---

**文件中被忽略的内容**：
- ROCm 相关：无（此文件不涉及）
- Backward 相关：无（此文件仅处理前向传播的张量转换，不涉及自动微分）
