## c10/util/irange.h 功能分析

这个文件实现了一个**整数范围迭代器**，提供类似 Python `range()` 的功能，用于在 C++ 中进行整数范围循环。

### 核心组件

**integer_iterator** (第16-73行)
- 模板化的整数迭代器，包装整数值
- 支持 `++` 前置和后置递增操作
- 特殊的相等比较逻辑：`one_sided` 模板参数为 `true` 时，任何负数结尾都会导致比较失败，用于处理空范围情况

**integer_range** (第77-95行)
- 容器适配器，保存 `begin` 和 `end` 迭代器
- 提供标准的 `begin()` 和 `end()` 方法，支持范围 for 循环

**irange() 函数** (两个重载)
1. **两参数版本** (第106-112行)：`irange(begin, end)` 创建半开区间 [begin, end)，如果 end ≤ begin 则返回空范围
2. **单参数版本** (第119-121行)：`irange(end)` 创建 [0, end) 范围，使用 `one_sided=true`

### 使用示例
```cpp
for (auto i : c10::irange(5)) {      // 0, 1, 2, 3, 4
  // ...
}
for (auto i : c10::irange(2, 10)) {  // 2, 3, ..., 9
  // ...
}
```

### 主要特性

- 支持任意整数类型（int, size_t, int64_t 等）
- 完全 constexpr，可在编译时使用
- 类型安全的整数混合（begin/end 类型不同时自动转换为 end 的类型）
- 特殊处理负范围和空范围

---

• 提供 Python 风格的整数范围迭代
• 支持 C++ 范围 for 循环
• 完全 constexpr 实现
• 处理空范围和负数边界情况
• 类型安全的跨类型范围构造
