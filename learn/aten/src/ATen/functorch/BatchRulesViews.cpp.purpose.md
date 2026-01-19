这个文件实现了 PyTorch functorch 库中 vmap（向量化映射）对各种视图操作的批处理规则（batch rules）。

## 核心概念

**批处理规则（Batch Rule）**：定义了当张量有额外的批次维度（batch dimension, bdim）时，如何执行操作。批次维度是被 vmap 映射的维度。

**基本模式**：
- 输入：原始张量 + 可选的批次维度位置（`std::optional<int64_t> bdim`）
- 输出：结果张量 + 新的批次维度位置
- 通用策略：使用 `moveBatchDimToFront` 将批次维度移到最前面，调整逻辑维度索引，执行操作

## 主要实现的视图操作

**维度操作**：
- `unsqueeze_batch_rule`: 在指定位置插入大小为1的维度，需要调整 dim 参数跳过批次维度
- `squeeze_batch_rule`: 移除所有大小为1的维度，但保留批次维度，追踪批次维度的新位置
- `squeeze_dim_batch_rule` / `squeeze_dims_batch_rule`: 移除指定维度，调整批次维度索引

**形状变换**：
- `view_batching_rule`: 重塑张量形状，在目标形状前插入批次大小
- `view_copy_batch_rule`: view 的拷贝版本
- `_unsafe_view_batch_rule`: 不安全的视图操作，如果视图无效则拷贝
- `_reshape_alias_batch_rule`: reshape 的别名版本

**索引和切片**：
- `select_batching_rule`: 在指定维度选择单个索引，将物理维度调整为跳过批次维度
- `slice_batch_rule`: 切片操作，调整 dim 参数
- `diagonal_batching_rule`: 提取对角线元素

**维度重排**：
- `transpose_int_batch_rule`: 转置两个维度，特殊处理标量张量情况
- `permute_batching_rule`: 任意维度排列，在排列列表前插入批次维度
- `roll_batch_rule`: 循环移位元素，调整 dims 参数或展平非批次维度

**扩展操作**：
- `expand_batch_rule`: 广播张量到更大形状，先 view 成正确维度数再 expand
- `repeat_batch_rule`: 重复张量元素，在 sizes 前插入1

**其他视图**：
- `flip_batch_rule`: 翻转指定维度
- `unfold_batch_rule`: 展开维度创建滑动窗口
- `narrow_copy_batch_rule`: 窄化拷贝
- `unsafe_split_batch_rule`: 不安全的分割操作
- `diag_embed_batch_rule`: 嵌入对角矩阵

**特殊操作**：
- `resize__plumbing`: 原地调整大小，需要更新 BatchedTensorImpl 的元数据
- `trace_decomp`: 矩阵迹的分解实现（`diagonal().sum()`）
- `tril_batch_rule` / `triu_batch_rule`: 下/上三角矩阵

## 注册机制

文件末尾通过 `TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m)` 将所有批处理规则注册到 FuncTorchBatched 调度键，使用 `VMAP_SUPPORT` 宏简化注册。

---

**Backward 相关内容**：
- `diagonal_backward_batch_rule`: 对角线操作的反向传播
- `select_backward_batch_rule`: select 操作的反向传播  
- `slice_backward_batch_rule`: slice 操作的反向传播

**ROCm 相关**：文件中未涉及 ROCm 特定内容
