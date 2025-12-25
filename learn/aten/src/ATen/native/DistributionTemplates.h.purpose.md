# DistributionTemplates.h 核心功能分析

这是 PyTorch 中用于实现各种概率分布采样的模板函数库，提供统一的接口和实现框架。

## 核心设计模式

文件采用 **模板方法模式**，定义了分布采样的通用流程：
1. 参数验证
2. 空张量快速返回
3. 创建 TensorIterator
4. 调用特定分布的 kernel

通过模板参数 `template<typename> class XXX_kernel` 和 `typename RNG`，将具体实现延迟到 CPU/CUDA 特化版本。

## 主要功能模块

### 1. Random (均匀整数分布)

**核心问题**：浮点类型精度导致的边界溢出

- `update_from()` / `update_to()`：调整 `[from, to)` 区间边界
  - **问题场景**：`random_(0, 65504)` 对于 `torch::half`，uint64 生成的 65503 转型后变成 65504，违反 `< to` 约束
  - **解决方案**：将边界向内收缩到下一个安全值。对于 65504，左移 32 变成 65472
  - 计算公式：`1LL << (n - std::numeric_limits<scalar_t>::digits + 1)`

**实现函数**：
- `random_impl()`：无参数版本，全范围随机
- `random_from_to_impl()`：三种模式
  - `[from, to)`：指定区间
  - `[from, max]`：from 到类型最大值
  - `[int64_min, int64_max]`：完整 2^64 范围

**边界检查**：`check_from_to_in_range()` 根据数据类型分发：
- 浮点型：检查是否在 `[lowest, max]`，警告超出精度范围（±2^digits）
- uint64：特殊处理避免类型转换陷阱
- 整数型：检查类型范围

### 2. Normal (正态分布)

**复数处理**：
- 将复数张量视为实数（`view_as_real`）
- 方差减半：`std/(std::sqrt(2))`，因为实部虚部各自的方差是总方差的一半

**三种重载形式**：
```cpp
normal(mean_tensor, std_scalar)  // 均值张量 + 标量标准差
normal(mean_scalar, std_tensor)  // 标量均值 + 标准差张量
normal(mean_tensor, std_tensor)  // 都是张量
```

**实现策略**：先生成 N(0, 1)，再通过 `output.mul_(std).add_(mean)` 变换

**CUDA 特殊注意**：注释中指出之前的 `addcmul_out` 实现有 bug，第三个参数不是 const 引用导致输出被覆盖

### 3. Uniform (连续均匀分布)

**复数处理**：递归调用实部虚部的浮点张量

**边界处理**：
- 检查 `[from, to]` 在数据类型范围内
- 检查 `to - from <= max` 避免溢出
- 自动截断到有效范围：`from = min(max(from, min), max)`

### 4. 其他分布

| 分布 | 参数约束 | 特殊处理 |
|------|---------|---------|
| LogNormal | std > 0 | 无 |
| Geometric | 0 < p < 1 | 无 |
| Exponential | lambda > 0 | 无 |
| Cauchy | sigma > 0 | 仅支持浮点类型 |
| Bernoulli | 0 ≤ p ≤ 1 | 支持标量/张量概率参数 |

## 关键宏定义

```cpp
CHECK_EMPTY_AND_RETURN(tensor)  // 空张量早返回优化（issue #103418）
CHECK_OUT_OF_BOUNDS             // 边界检查 + 错误抛出
WARN_OUT_OF_BOUNDS              // 精度警告（未来版本会升级为错误）
CHECK_NORMAL_STD                // 标准差非负检查
CHECK_NORMAL_TENSOR_STD         // 张量标准差：非复数 + 所有元素 ≥ 0
```

## 技术要点

1. **类型分发**：`AT_DISPATCH_FLOATING_TYPES_AND2` / `AT_DISPATCH_V2` 实现编译期类型选择
2. **内存重叠检查**：Bernoulli 中调用 `assert_no_internal_overlap(self)`
3. **名称推断**：使用 `NoNamesGuard` 和 `propagate_names` 处理命名张量
4. **输出调整**：`resize_output()` 根据广播规则推断输出形状
5. **性能优化**：`borrowing_nullary_op()` 避免不必要的内存分配

---

**其他相关内容**：
- 未涉及 ROCm 特定优化（文件中无 ROCm 代码）
- 未涉及反向传播（这些是采样操作，一般不可导或使用重参数化技巧在更高层实现）
