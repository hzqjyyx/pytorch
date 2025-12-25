## LegacyBatchedFallback 主要功能

这个文件实现了 PyTorch vmap（vectorized map）的回退机制，用于处理那些没有专门批处理规则的算子。

### 核心思想

当一个算子没有实现批处理规则时，fallback 会：
1. 将所有 BatchedTensor 输入按批次维度切片
2. 对每个切片独立调用原始算子
3. 将所有输出切片通过 `stack` 操作组合成最终结果

### 关键函数

**`batchedTensorForLoopFallback`** (主入口)
- 处理非原地操作
- 检查算子是否可以使用 fallback（不支持 out= 参数、视图操作、TensorList 参数）
- 要求所有返回值都是 Tensor
- 使用循环遍历所有批次，收集输出切片

**`batchedTensorInplaceForLoopFallback`** (原地操作专用)
- 处理原地修改的算子（如 `add_`）
- 执行额外的 vmap 兼容性检查：确保 `self` 参数的 vmap 层级包含所有其他参数的层级
- 如果某个参数在 `self` 未参与的 vmap 层级上被批处理，则报错（因为元素数量不匹配）

### 算法流程

1. **识别 BatchedTensor**
   - 遍历所有参数，找出 BatchedTensor
   - 记录它们在参数列表中的位置

2. **物理视图转换**
   - 使用 `MultiBatchVmapTransform::logicalToPhysical` 将批次维度移到最前面
   - 计算总批次数：`num_batches = product(batch_sizes)`

3. **循环执行**
   - 对每个线性索引 `linear_idx`：
     - 使用 `computeIndex` 转换为多维索引
     - 对 BatchedTensor 参数进行切片，非批处理参数保持不变
     - 调用原始算子
     - 收集输出（非原地操作）或丢弃返回值（原地操作）

4. **组合结果**（仅非原地操作）
   - 使用 `safeStack` 将每个返回值的所有切片堆叠
   - 通过 `PhysicalToLogicalMap` 将批次维度转换回逻辑位置

### 辅助功能

**`computeIndex`**
- 将线性索引转换为多维索引
- 例：`linear_idx=3, sizes=[5,2]` → `[1,0]`

**`safeStack`**
- 安全地堆叠 Tensor 列表
- 处理 undefined grad 的特殊情况（vmap through backward）
- 要么全部定义，要么全部未定义，不支持混合

**`isInplaceOp`**
- 判断是否为原地操作：
  - 第一个参数是可变 Tensor 且被写入
  - 第一个参数被返回
  - 其他参数没有别名信息

### 限制和检查

- 不支持批次大小为 0 的维度
- 不支持 out= 参数和视图操作
- 不支持 TensorList 参数
- 原地操作有严格的 vmap 层级兼容性要求
- 性能较差（额外的拷贝开销），建议为常用算子编写专门的批处理规则

### 警告机制

`warnFallback` 在使用 fallback 时发出警告，提示：
- 性能下降
- 建议使用新版 `torch.vmap` 和 `torch.func.*` API 替代旧的 `torch._vmap_internals.vmap`

---

**与 ROCm/Backward 相关的内容：**
- NOTE [vmap through backward and undefined grad]：处理反向传播中某些样本梯度为 undefined 的情况
- NOTE [vmap-incompatible in-place operations]：解释原地操作的 vmap 兼容性约束
