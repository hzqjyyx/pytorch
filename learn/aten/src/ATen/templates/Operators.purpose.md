这两个文件是 ATen 操作符代码生成的模板文件。

**Operators.h** 的主要功能：

- 定义了 `ATEN_FN` 和 `ATEN_FN2` 宏，用于获取不重载的函数版本的操作符
- 定义了 `ATEN_OP` 和 `ATEN_OP2` 宏，用于访问操作符的编译时元数据（名称、重载名称、schema 等）
- 提供了条件编译检查，防止在不适当的地方包含此文件（如 `TORCH_ASSERT_NO_OPERATORS` 和 `TORCH_ASSERT_ONLY_METHOD_OPERATORS`）
- 包含必要的头文件（SymInt、Scalar、TensorOptions 等）
- 声明了 `at::_ops` 命名空间中的所有操作符相关类和函数

**Operators.cpp** 的主要功能：

- 包含 Tensor 和 Dispatcher 头文件
- 根据编译选项（`AT_PER_OPERATOR_HEADERS`）有条件地包含操作符头文件
- 在 `at::_ops` 命名空间中定义所有操作符的实现

**核心要点：**

• 这是生成式代码的模板（`${...}` 占位符会被代码生成器替换）
• 提供了访问 ATen 操作符的统一、类型安全的 API
• 支持两种编译模式：单体头文件 vs 按操作符分离头文件
• 宏接口隐藏了 `_ops` 命名空间的实现细节，提供了稳定的公共 API
