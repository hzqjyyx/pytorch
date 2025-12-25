# BatchRulesHelper 核心功能

## 1. 批次维度操作

### moveBatchDimToFront
将批次维度移动到张量的第 0 维。如果批次维度已经在第 0 位或不存在，直接返回原张量。

### rankWithoutBatchDim / numelWithoutBatchDim
计算去除批次维度后的逻辑秩和元素数量。这些函数用于理解张量的"逻辑形状"（用户视角）而非"物理形状"（带批次维度）。

### getPhysicalDim / getPhysicalDims
将逻辑维度索引转换为物理维度索引。假设批次维度位于第 0 位，逻辑维度需要 +1 才能得到物理维度。

```cpp
// 示例：tensor 形状 [B, 3, 4, 5]，逻辑上是 [3, 4, 5]
// logical_dim=1 -> physical_dim=2
```

## 2. 维度重塑工具

### reshape_dim_into
将 `src` 维度合并到 `dst` 维度中。先将 `src` 移动到 `dst` 位置，然后重塑使两个维度相乘。

```cpp
// [2, 3, 4, 5] reshape_dim_into(1, 2) -> [2, 4*3, 5] = [2, 12, 5]
```

### reshape_dim_outof / reshape_dim_outof_symint
将一个维度拆分成两个维度。在 `src` 位置插入新维度 `size1`，原维度变为 `size2 = original_size / size1`。

```cpp
// [2, 12, 5] reshape_dim_outof(1, 3) -> [2, 3, 4, 5]
```

## 3. 广播对齐

### maybePadToLogicalRank
为带批次维度的张量在批次维度后插入大小为 1 的维度，使其逻辑秩达到指定值。用于广播前对齐张量形状。

```cpp
// tensor [B, 3], logical_rank=3 -> [B, 1, 1, 3]
```

### _binary_pointwise_helper
处理二元逐点操作的核心辅助函数：
1. 计算两个张量的最大逻辑秩
2. 将批次维度移到前面
3. 处理 (0D, ND) 情况下的类型提升
4. 填充维度使两个张量逻辑秩相同

### ensure_has_bdim
确保张量有批次维度。如果没有，通过 `expand` 在前面添加批次维度。

## 4. 批处理规则宏

### BASIC_UNARY_BATCH_RULE
一元操作的默认批处理规则：直接调用原函数，批次维度位置不变。

### VARIADIC_BDIMS_BATCH_RULE
可变批次维度规则：将所有输入的批次维度移到第 0 位，调用函数后输出批次维度固定在第 0 位。

### EXISTING_BDIM_BATCH_RULE
假设输入有批次维度，将其 reshape 到第 0 维，调用函数后再 reshape 回来。

### Boxed 批处理规则

#### boxed_tensor_inputs_batch_rule
通用的装箱批处理规则框架：
- 从栈中提取所有张量参数及其批次维度
- 应用自定义处理函数（如 `handle_pointwise_ops`）
- 调用原操作
- 将返回值包装回批处理张量

#### boxed_existing_bdim_all_batch_rule
适用于所有张量都有批次维度的情况：
- 对所有张量调用 `ensure_has_bdim` 和 `reshape_dim_into`
- 执行操作
- 用 `reshape_dim_outof` 恢复批次维度

#### boxed_all_tensors_have_optional_bdim
处理 NN 算子的特殊规则（如卷积）：
- 判断是否为"无批次维度情况"（logical_rank == feature_rank）
- 无批次：直接移动批次维度到前面
- 有批次：reshape 批次维度到第 0 维
- 支持指定某个张量需要连续化

## 5. 辅助工具

### valIfNonempty
条件值传递：如果第一个 optional 有值，返回 `new_val`，否则返回 nullopt。

### check_randomness
验证随机操作的合法性：
- 不允许在 `RandomnessType::Error` 模式下调用
- 不允许在 `RandomnessType::Same` 模式下对批处理张量操作

### vmapIncompatibleInplaceError
用于就地操作不兼容时抛出详细错误信息。

### get_bdim_size2/3/4
从多个张量中找到第一个存在的批次维度大小。

### range
生成连续整数向量 `[start, stop)`。

---

**相关但忽略的内容：**
- ROCm 特定实现
- Backward pass / 自动微分相关逻辑
- 梯度处理
