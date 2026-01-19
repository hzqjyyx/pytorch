这个文件实现了 PyTorch functorch 库中的**遗留（已弃用）批处理规则（batching rules）**，用于支持 `vmap` 操作。

## 核心概念

**批处理规则**定义了当张量具有额外的批次维度时，如何调用算子。当使用 `vmap` 时，被映射的维度会被记录为批次维度，批处理规则负责将逻辑操作转换为物理操作。

基本流程：
1. 将逻辑 BatchedTensor 转换为物理张量的视图
2. 将逻辑参数（维度索引、形状）转换为对应物理参数
3. 在物理张量上调用 at:: 操作
4. 将物理结果转换回 BatchedTensor

## 主要实现的批处理规则

### 就地形状操作
- **`squeeze_`/`squeeze_dim_`/`squeeze_dims_`**: 移除大小为1的维度，需要跟踪和调整批次维度位置
- **`unsqueeze_`**: 添加大小为1的维度，根据插入位置调整批次维度索引
- **`transpose_`**: 转置两个维度，处理批次维度的映射关系

### 张量分割/组合
- **`split`/`split_with_sizes`**: 将张量沿指定维度分割，转换维度索引并应用物理到逻辑的映射
- **`unbind`**: 沿维度解绑张量，返回张量列表
- **`cat`**: 连接张量列表，处理混合批次/非批次张量，确保批次维度对齐
- **`stack`**: 堆叠张量，在新维度上组合
- **`block_diag`**: 创建块对角矩阵，通过循环处理每个批次

### 高级视图操作
- **`as_strided`**: 创建自定义步幅视图，包含复杂的内存安全检查（lines 288-489 详细证明了为什么这个操作是安全的）
- **`new_empty_strided`**: 创建具有自定义步幅的新张量，计算物理步幅以保持批次维度的连续性

### 嵌套张量支持
- **`nested_cat_batching_rule`**: 处理嵌套张量的连接，解绑每个嵌套张量后逐组连接

## 辅助函数

- `participatesInCurrentLevel()`: 检查张量是否参与当前 vmap 层级
- `get_current_level()`: 获取当前动态层 ID
- `is_allowed_dim_on_scalar_tensor()`: 处理标量张量的特殊维度规则（0 和 -1）
- `maximum_indexable_location()`: 计算给定形状/步幅的最大可索引位置
- `checkBasicAsStridedValidForSlice()`: 验证 as_strided 操作的内存安全性

## 注册机制

使用 `TORCH_LIBRARY_IMPL` 宏注册批处理规则：
- `FuncTorchBatched` 调度键：注册标准批处理规则和回退函数
- `BatchedNestedTensor` 调度键：注册嵌套张量特定规则和回退函数

## 关键设计点

- **维度映射**: 逻辑维度需要根据批次维度位置调整为物理维度
- **层级检查**: 只处理参与当前 vmap 层级的张量，其他张量直接调用原始操作
- **元数据刷新**: 就地操作后需要调用 `refreshTensorMetadata()` 更新 BatchedTensor 的元数据
- **内存安全**: `as_strided` 包含详尽的边界检查，确保批处理后的视图不会访问越界内存

---

**ROCm 相关**: 无

**Backward 相关**: 无（此文件专注于前向传播的批处理规则）
