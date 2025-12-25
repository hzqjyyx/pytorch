# TensorAdvancedIndexing 文件分析

这两个文件实现了 PyTorch 中张量的高级索引功能，对应 NumPy 的 "advanced indexing"。

## 核心功能

### 1. 主要操作

**index** - 通过索引张量获取元素：
- 支持 Long/Bool/Byte 类型的索引张量
- Bool/Byte 张量会通过 `nonzero()` 展开为 Long 张量
- 所有索引一起广播并作为一个整体迭代
- 实现公式：`result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M], ...]`

**index_put_** - 通过索引设置元素值：
- 支持 `accumulate` 参数用于累加而非替换
- 有两种实现路径：
  - 高效路径：将索引视为逐元素操作
  - accumulate 路径：组合索引为单一线性索引，使用 `put_`

### 2. 索引操作的高效实现

**核心思路**（针对非 accumulate 情况）：

```
步骤 1: 广播所有索引张量到统一形状
步骤 2: 记录每个被索引维度的 stride
步骤 3: 用 stride=0 替换被索引子空间
步骤 4: 为索引张量添加 size=1 的维度以匹配结果形状
```

实际计算：
```c
result[...] = *(&x[...] + ind_1[...] * x.stride(1) + ind_2[...] * x.stride(2) + ...)
```

**关键数据结构** - `AdvancedIndex`：
- `src`: 重新设置 stride 的源张量
- `indices`: 重塑后的索引张量列表
- `indexed_sizes/indexed_strides`: 被索引维度的原始大小和步长
- `dims_before/dims_after`: 索引子空间前后的维度数

**特殊优化**：
- CUDA/MPS/XPU 上强制所有索引张量使用相同 stride（通过 `contiguous()`）
- 非相邻索引会先转置张量

### 3. gather/scatter 系列操作

**gather** - 沿指定维度收集元素：
- 输出形状与 index 相同
- `out[i][j][k] = input[index[i][j][k]][j][k]` (dim=0 时)
- 支持 `sparse_grad` 参数

**scatter** 及其变体：
- `scatter`: 沿维度分散写入，支持 reduce 操作（已废弃，建议用 `scatter_reduce`）
- `scatter_add`: 累加写入
- `scatter_reduce`: 支持新的 reduce 选项（prod/mean/amax/amin）

**展开索引的优化版本**：
- `scatter_add_expanded_index_stub`
- `scatter_reduce_expanded_index_stub`
- `gather_expanded_index_stub`

### 4. index_select/index_copy/index_add 操作

**index_select** - 沿维度选择索引指定的切片：
- CPU 实现有多路径优化：
  - `dim=1` 且连续时的快速路径
  - 批量 memcpy 优化（非浮点或 block_size > 1）
  - 单精度浮点特殊优化
- 支持量化张量（仅 per-tensor affine）

**index_copy** - 将 source 复制到 index 指定位置：
- 确定性算法模式下使用 `index_put_` 实现
- 使用重新 stride 的迭代器避免维度 `dim` 上的前进

**index_add** - 沿维度累加索引位置的值：
- CPU 实现针对 `dim=0` 或 `dim=self.dim()-1` 时优化：
  - 条件满足时转换为 `scatter_add_` 调用
  - 检查除 dim 外的维度是否匹配（不支持广播）
- 多维情况复用二元操作迭代器
- 支持 alpha 缩放参数

### 5. index_reduce 操作

支持的 reduce 模式：
- `prod`: 乘积
- `mean`: 平均值（需要计算 counts）
- `amax`: 最大值
- `amin`: 最小值

实现特点：
- `include_self=false` 时初始化为单位元（prod→1, max→-inf, min→+inf, mean→0）
- mean 操作需额外维护计数张量，最后执行除法
- NaN 传播：min/max 遇到 NaN 直接返回 NaN

### 6. put/take 操作

**put_** - 按扁平化索引放置值：
- index 必须是 Long 类型
- source 和 index 元素数必须相同
- accumulate 模式在 CUDA 上使用原子操作（非确定性）

**take** - 按扁平化索引提取值：
- 将张量视为一维进行索引
- 输出形状与 index 相同

### 7. masked_fill/masked_scatter/masked_select

这些操作通过 dispatch stub 分发到不同后端实现。

### 8. 不安全索引操作

**_unsafe_index** - 跳过布尔索引（避免动态形状）：
- 仅允许 Long/Int 索引
- 委托给标准 `index`

**_unsafe_masked_index** - 条件索引：
- 等价于 `where(mask, self[indices], fill)`
- mask=false 时不执行索引（允许越界）
- 通过 clamp 将索引限制在有效范围

**_unsafe_masked_index_put_accumulate** - 对应的反向操作：
- masked_value 在 mask=false 处填充 0
- 委托给 `_unsafe_index_put`

### 9. 量化张量支持

**quantized_index**：
- 仅支持 per-tensor affine/symmetric 量化
- 实现：dequantize → index → quantize
- 保持相同的 scale 和 zero_point

## 调度机制

使用 `DECLARE_DISPATCH` 宏定义的 dispatch stubs：
- `index_stub`, `index_fill_stub`, `index_copy_stub`
- `index_put_stub`, `index_put_with_sort_stub`
- `put_stub`, `take_stub`
- `masked_fill_stub`, `masked_select_stub`, `masked_scatter_stub`
- `gather_stub`, `scatter_stub`, `scatter_fill_stub`, `scatter_add_stub`
- `scatter_reduce_stub`, `scatter_scalar_reduce_stub`, `scatter_reduce_two_stub`

CPU 后端没有注册 `index_put_with_sort_stub`（仅 CUDA/XPU）。

## 内存重叠检查

所有操作都进行严格的内存重叠检查：
- `assert_no_internal_overlap`: 检查张量自身重叠
- `assert_no_overlap`: 检查不同张量间重叠
- `assert_no_partial_overlap`: 检查部分重叠

扩展张量上使用 `index_put_` 会触发警告。

## 确定性算法支持

CUDA 上当 `globalContext().deterministicAlgorithms()=true` 时：
- `index_copy` 使用 `index_put_` 实现
- `index_put_` 强制使用 `index_put_with_sort_stub`

---

**ROCm 相关**：文件中未直接涉及 ROCm 特定代码，ROCm 应通过 dispatch 机制使用相同接口

**Backward 操作**：
- `index_select_backward`: 使用 `new_zeros + index_add` 实现梯度传播
- `gather_backward`: 在 ops 头文件中声明
- `masked_select_backward`: 在 ops 头文件中声明
