- **数值类型包装**：Scalar 类将各种 C++ 数值类型（int64_t、double、complex、bool、uint64_t 等）统一包装成单一类型，便于跨类型的统一操作

- **隐式类型转换**：通过大量重载构造函数支持 C++ 数值字面量隐式转换到 Scalar，使得 API 可以同时提供 `add(Tensor)` 和 `add(Scalar)` 两种重载

- **运行时类型判断**：提供 `isFloatingPoint()、isIntegral()、isComplex()、isBoolean()` 等方法在运行时检查标量数据类型

- **类型转换访问器**：通过 `toDouble()、toLong()、toUInt64()` 等方法将标量值转换为特定类型，支持范围检查与溢出检测

- **符号类型支持**：支持 SymInt、SymFloat、SymBool 等符号计算类型，用于动态形状推导和符号执行

- **联合体存储**：使用 union 类型高效存储不同类型的数值，减少内存开销

- **0维张量表示**：逻辑上代表单个元素的 0 维张量，与 Tensor 对象的接口兼容性好，但数据结构更轻量

- **相等比较**：`equal()` 模板方法支持与 C++ 原生数值类型的精确比较

- **移动语义与内存管理**：实现 move 语义优化，对符号类型使用 intrusive_ptr 进行引用计数管理
