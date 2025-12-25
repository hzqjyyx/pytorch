# Unique.cpp 功能解析

这个文件实现了 PyTorch 中查找张量唯一元素的功能，包含多个变体的 unique 操作。

## 核心实现

### 1. 布尔类型特化 (`unique_cpu_bool_template`)
**位置**: 35-107行

专门优化布尔张量的 unique 操作。由于布尔值只有 true/false 两种，采用计数归约而非排序：

- 并行统计 true 的数量（58-66行）
- 根据 true/false 出现情况构建输出（最多2个元素）
- 如需 inverse_indices，并行映射每个元素到输出索引（96-105行）

### 2. 基于排序的 unique (`unique_cpu_sorted_template`)
**位置**: 158-264行

通用的 unique 实现，适用于所有数值类型：

**算法流程**:
1. 展平并排序输入（183-185行）
2. 并行识别唯一元素：通过 `is_unique` 函数判断当前元素是否与前一个不同（197-204行）
3. 线程间协调：每个线程计算局部唯一数量，然后计算全局偏移量（206-212行）
4. 并行生成结果（234-253行）：
   - 写入唯一值到 output
   - 通过排序后的索引逆向映射生成 inverse_indices
   - 记录每个唯一值的首次出现位置
5. 如需 counts，通过差分计算每个唯一值的出现次数（255-262行）

**NaN 处理**:
- `IsUnique` 模板（122-140行）支持两种 NaN 比较策略
- `equal_nan=true`: 所有 NaN 视为相同
- `equal_nan=false`: 每个 NaN 都是唯一的（默认未启用）

### 3. 连续 unique (`unique_consecutive_cpu_template`)
**位置**: 267-322行

不排序，只去除连续重复元素：

- 单遍扫描输入（299-312行）
- 当前值与上一个值不同时写入输出
- 同时更新 inverse_indices 和 counts

**性能特点**: 时间复杂度 O(n)，但只处理连续重复

### 4. 按维度 unique (`_unique_dim_cpu_template`)
**位置**: 357-437行

沿指定维度查找唯一切片：

**处理流程**:
1. 将目标维度移到第0维，重塑为 `[dim_size, -1]`（385-387行）
2. 如非 consecutive 模式，按字典序排序切片（395-408行）
3. 使用 `_unique_dim_cpu_impl`（324-354行）去重：
   - 逐个比较张量切片（通过 `at::equal`）
   - 生成 inverse_indices 和 counts
4. 恢复原始形状和维度顺序（429-434行）

**辅助函数 `_unique_dim_cpu_impl`**:
- 使用迭代器范式处理 Tensor 向量
- 通过 `at::equal` 比较整个张量切片而非标量

## 公共接口

### `_unique_cpu` (442-455行)
返回 (output, inverse_indices)

### `_unique2_cpu` (458-468行)  
返回 (output, inverse_indices, counts)

### `unique_dim_cpu` (471-476行)
沿指定维度的 unique（需排序）

### `unique_dim_consecutive_cpu` (479-483行)
沿指定维度的连续 unique

### `unique_consecutive_cpu` (486-493行)
- 无维度参数时调用标量版本
- 有维度参数时分派到 `unique_dim_consecutive_cpu`

## 性能优化要点

- **并行化**: 使用 `at::parallel_for` 在统计、写入、计数计算等阶段并行处理（grain_size = GRAIN_SIZE）
- **类型分派**: `AT_DISPATCH_V2` 为不同数据类型生成特化代码
- **内存局部性**: 通过 `c10::load` 优化数据加载
- **bool 特化**: 避免不必要的排序，直接计数
- **融合计算**: inverse_indices 和 counts 在同一循环中生成，几乎零开销

## 其他要点

- **边界情况**: 处理空张量、零维度张量（369-382行）
- **内存布局**: 要求输入 contiguous（通过 `.contiguous()` 确保）
- **索引映射**: inverse_indices 通过排序索引的逆映射还原到原始位置（249-250行）
