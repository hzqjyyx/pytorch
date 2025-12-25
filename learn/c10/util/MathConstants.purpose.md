## MathConstants.h 和 MathConstants.cpp 的功能分析

**MathConstants.h** 定义了一组数学常数的模板化实现，支持多种数据类型（包括 `double`、`float`、`BFloat16`、`Half` 等）：

- 在 `c10::detail` 命名空间中定义了模板函数，返回各种数学常数的精确值
- 在外层 `c10` 命名空间中通过模板变量暴露这些常数供外部使用
- 针对 `BFloat16` 和 `Half` 两种低精度浮点类型提供了特化实现，使用比特模式直接构造

**MathConstants.cpp** 是编译时验证文件：

- 为 MSVC 编译器启用数学常数定义（`_USE_MATH_DEFINES`）
- 通过 `static_assert` 验证 `c10::pi<double>` 和 `c10::frac_sqrt_2<double>` 与标准库的 `M_PI` 和 `M_SQRT1_2` 完全相等

---

### 核心功能总结：

- **定义数学常数**：π、e、黄金比例、ln(2)、ln(10) 等 13 个数学常数
- **泛型设计**：使用 C++ 模板支持任意数据类型
- **低精度支持**：为 `BFloat16` 和 `Half` 提供特化实现
- **编译时验证**：确保常数与标准库值一致
- **GPU 兼容性**：`C10_HOST_DEVICE` 宏使常数可在 CPU/GPU 代码中使用
