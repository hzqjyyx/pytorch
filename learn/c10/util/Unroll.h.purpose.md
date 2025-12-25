# `c10/util/Unroll.h` 功能分析

这个文件提供了一个编译时循环展开的实用工具。

## 核心机制

**`ForcedUnroll<n>` 模板结构** 通过递归模板特化实现循环展开：

1. **通用模板** (`ForcedUnroll<n>`): 递归调用 `ForcedUnroll<n-1>`，然后执行 `f(integral_constant<int, n-1>, args...)`
2. **特化模板** (`ForcedUnroll<1>`): 基础情况，执行 `f(integral_constant<int, 0>, args...)`

这种设计在编译时展开整个循环，每次迭代传递一个 `std::integral_constant` 作为索引。

## 使用示例

```cpp
c10::ForcedUnroll<4>{}(f);
// 展开后等价于：
// f(integral_constant<int, 0>{});
// f(integral_constant<int, 1>{});
// f(integral_constant<int, 2>{});
// f(integral_constant<int, 3>{});
```

## 关键特点

- **`C10_ALWAYS_INLINE` 宏**: 强制内联，确保编译器真正展开代码
- **支持可变参数**: 可传递额外的 `args` 给函数对象
- **编译时计算**: 循环边界必须在编译时已知
- **跨编译器可移植**: 比编译器特定的 pragma 更可靠

## 主要用途

- **性能优化**: 避免运行时循环开销，特别适合小固定循环
- **模板元编程**: 在编译时对编译期常数进行迭代操作
- **SIMD 和向量化**: 为向量化操作展开循环

---

**核心特点：**
- 编译时循环展开工具
- 递归模板特化实现
- 强制内联确保代码展开
- 支持可变参数传递
- 适用于编译期已知的循环边界
