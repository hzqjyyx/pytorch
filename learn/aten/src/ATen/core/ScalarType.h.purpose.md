- **数据类型枚举定义**：定义了 `ScalarType` 枚举，包含 45 种数据类型（整数、浮点、复数、量化、Float8、低精度等）

- **类型映射机制**：通过 `ScalarTypeToCPPType` 和 `CppTypeToScalarType` 模板实现编译时的 ScalarType 与 C++ 类型相互映射

- **宏定义集合**：提供多个宏（`AT_FORALL_SCALAR_TYPES_*`）用于遍历不同类别的数据类型，支持模板实例化和代码生成

- **类型检查函数**：判断类型属性的内联函数
  - `isIntegralType()` - 整数类型判断
  - `isFloatingType()` - 浮点类型判断
  - `isComplexType()` - 复数类型判断
  - `isQIntType()` - 量化类型判断
  - `isFloat8Type()` - Float8 类型判断

- **类型转换函数**：
  - `toQIntType()` - 转换为量化类型
  - `toUnderlying()` - 获取量化类型的基础类型
  - `toRealValueType()` - 复数转实数
  - `toComplexType()` - 实数转复数
  - `canCast()` - 检查类型转换合法性

- **工具函数**：
  - `toString()` - 类型转字符串
  - `elementSize()` - 获取元素大小
  - `isSignedType()` - 有符号判断
  - `getDtypeNames()` / `getStringToDtypeMap()` - 类型名称/映射
