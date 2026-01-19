## InferSize.h 文件分析

这个文件实现了张量形状推断的核心逻辑，主要用于处理包含 `-1` 的形状规范。

**核心函数：**

- **`infer_size_impl()`** - 通用模板函数，推断形状中 `-1` 维度的实际大小
  - 遍历形状数组，计算已知维度的乘积
  - 检查是否恰好有一个 `-1` 维度（多于一个会报错）
  - 验证总元素数是否匹配：`numel = newsize × inferred_dim`
  - 计算推断维度：`inferred_dim = numel / newsize`

- **`infer_size()`** - 处理 `IntArrayRef` 和 `int64_t`，返回 `std::vector<int64_t>`

- **`infer_size_dv()`** - 两个重载版本
  - 第一个处理 `IntArrayRef`，返回 `at::DimVector`
  - 第二个处理 `c10::SymIntArrayRef`，返回 `at::SymDimVector`（支持符号整数）

**关键特性：**

- 支持 NumPy 风格的 `-1` 自动推断（如 `view(-1, 10)` 自动计算第一维）
- 验证新形状与原张量元素总数的兼容性
- 处理边界情况（如 `view(-1, 0)` 会报错，因为无法推断）
- 通过模板支持多种数组类型和数值类型
- 支持符号维度（SymInt），用于动态形状推断

**使用场景：**

主要用于 `reshape()`、`view()` 等形状变换操作中，当用户指定 `-1` 时自动计算该维度大小。
