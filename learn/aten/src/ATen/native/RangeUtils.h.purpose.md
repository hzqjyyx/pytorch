# RangeUtils.h 功能分析

这个文件定义了用于计算 `arange` 操作输出大小的工具函数。

## 核心函数: `compute_arange_size`

**签名:**
```cpp
template <typename scalar_t>
int64_t compute_arange_size(const Scalar& start, const Scalar& end, const Scalar& step)
```

**功能流程:**

1. **类型转换** - 将输入的 `start`, `end`, `step` 转换为目标标量类型的累积类型 (`accscalar_t`)

2. **参数验证** - 三个检查:
   - `step` 不能为零
   - `start` 和 `end` 必须是有限值（非 NaN/Inf）
   - step 符号与范围方向一致（step > 0 时 end ≥ start；step < 0 时 end ≤ start）

3. **大小计算** - 分两种情况:
   - **int64_t**: 使用整数算术避免精度损失，公式为 `ceil((end - start + step - sgn) / step)`
   - **其他类型**: 转换为 double 精度计算，公式为 `ceil((end - start) / step)`
   - 使用 double 是为了保证 CPU/GPU 的一致性

4. **溢出检查** - 确保计算结果在 int64_t 范围内

---

## 要点总结

- 用于计算 arange 操作的元素个数
- 处理不同标量类型的精度差异
- 特殊处理 int64_t 以避免精度损失
- 包含多层验证防止非法范围参数
- 返回值为 int64_t 类型的元素个数
