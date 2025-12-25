**LegacyBatching.cpp - 功能分析**

该文件实现了 PyTorch vmap 功能中的批次维度管理，核心机制是将普通张量和批次张量之间进行转换。

**主要功能：**

- `_add_batch_dim()` - 为张量添加批次维度（out-of-place），封装调用 `addBatchDim()` 函数

- `has_level()` - 检查张量是否包含特定级别的批次维度，通过遍历 BatchedTensorImpl 中的批次维度列表实现

- `remove_existing_batch_dim()` - 从 BatchedTensor 中移除指定级别的批次维度，返回转换后的张量和维度索引
  - 如果只有一个批次维度，直接返回底层张量
  - 如果有多个批次维度，构建新的批次维度列表，计算原批次维度在新张量中的逻辑位置

- `maybe_movedim()` - 有条件的维度移动，如果源维度和目标维度相同则返回原张量，否则调用 `movedim()`

- `_remove_batch_dim()` - 主要 API，移除张量中指定级别的批次维度
  - 如果张量不包含该级别的批次维度，则通过 `expand()` 在指定位置插入该维度
  - 如果包含，先移除再移到目标维度位置

**核心用途：**

- 支持 vmap 的进出操作：入 vmap 时添加批次维度，出 vmap 时根据 `out_dims` 参数移除并重新定位批次维度
- 处理嵌套 vmap 场景，其中内层张量可能不与外层批次维度交互
