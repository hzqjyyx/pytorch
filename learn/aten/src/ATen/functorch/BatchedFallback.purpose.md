# BatchedFallback 功能分析

这两个文件实现了 PyTorch 的 vmap fallback 机制，用于在没有专门 batching rule 的情况下，为算子提供自动批处理支持。

## 核心机制

**Fallback 执行流程**：
1. 识别输入中的 BatchedTensor
2. 通过 `MultiBatchVmapTransform` 将所有批次维度移到张量前面
3. 对每个批次切片，逐个调用原始算子
4. 将结果切片 stack 起来形成最终输出

## 主要函数

### `batchedTensorForLoopFallback`
- **用途**：处理 out-of-place 操作的 fallback
- **限制**：
  - 仅支持返回值全是 Tensor 的算子
  - 不支持参数包含 TensorList 的算子
  - 不支持 out= 变体和 view 操作
  - 不支持 batch size 为 0 的情况
- **实现**：
  - 计算总批次数 `num_batches = product(batch_sizes)`
  - 用 `computeIndex` 将线性索引转换为多维索引
  - 循环遍历每个批次，对 BatchedTensor 参数进行索引切片
  - 将结果存储在 `output_shards`，按 `[a0, a1, ..., b0, b1, ..., c0, c1, ...]` 布局
  - 最后对每个输出使用 `safeStack` 合并切片

### `batchedTensorInplaceForLoopFallback`
- **用途**：处理 in-place 操作的 fallback
- **核心检查**：验证 vmap level 兼容性
  - 如果 `self` 不在某个 vmap level 上，但其他参数在，则报错
  - 原因：无法对元素数量不匹配的张量执行 in-place 操作
- **实现**：与 out-of-place 类似，但直接修改 `self`，最后返回 `self`

### `batchedNestedTensorForLoopFallback`
- **用途**：处理 NestedTensor 的 fallback
- **特点**：
  - 使用 `unbind()` 解绑嵌套张量的组件
  - 要求所有 batched 参数有相同数量的组件
  - 对每个组件独立调用算子
  - 用 `_nested_tensor_from_tensor_list` 重建 NestedTensor
  - 不支持 in-place 操作

### `vmapErrorFallback`
- 直接抛出错误，用于需要特殊处理但尚未实现 batching rule 的算子

## 辅助功能

**`computeIndex`**：
- 将线性索引转换为多维索引
- 例：`linear_idx=3, sizes=[5,2]` → `[1, 0]`

**`safeStack`**：
- 处理 backward 时可能出现的 undefined grad
- 全部 defined → 正常 stack
- 全部 undefined → 返回 undefined
- 部分 undefined → 报错

**`warnFallback`**：
- 检查 fallback 是否启用（测试用）
- 发出性能警告，提示用户提交 issue 实现专门的 batching rule

## 配置选项

- `isVmapFallbackEnabled` / `setVmapFallbackEnabled`：控制是否启用 fallback（默认启用）
- `isVmapFallbackWarningEnabled` / `setVmapFallbackWarningEnabled`：控制是否显示警告（默认启用）

## 性能考虑

Fallback 引入额外的拷贝开销（stacking 切片输出），性能不佳，因此优先为算子编写专门的 batching rule。

---

**ROCm 相关**：无

**Backward 相关**：
- `safeStack` 处理 backward 函数返回的 undefined grad
- NOTE [vmap through backward and undefined grad] 说明了全 undefined 时返回 undefined 的策略
