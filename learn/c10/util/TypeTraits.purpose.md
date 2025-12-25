## TypeTraits.h 文件解析

这个文件定义了一系列 C++ 类型特征（type traits）模板，用于编译时的类型检查和特性判断。

### 核心功能模块

**类型操作能力检测**
- `is_equality_comparable<T>`: 检查类型 T 是否支持 `==` 操作符
- `is_hashable<T>`: 检查类型 T 是否可被 `std::hash` 哈希
- 这两个都基于 SFINAE（替换失败不是错误）技术，通过 `std::void_t` 检测特定操作是否可编译

**函数类型识别**
- `is_function_type<T>`: 识别纯函数类型（如 `Result(Args...)`）
- `is_instantiation_of<Template, T>`: 检查 T 是否为 Template 的实例化（如 `vector<int>` 是 `vector` 的实例化）

**函数对象（Functor）检测**
- `is_functor<Functor>`: 判断类是否定义了 `operator()`
- 通过 `strip_class` 辅助模板移除 `operator()` 的 class 类型前缀，支持 const 和非 const 版本

**Lambda 表达式分析**
- `is_stateless_lambda<T>`: 判断 lambda 是否无状态（无捕获）
- 实现原理：C++ 标准规定无状态 lambda 可转换为函数指针，利用 `std::is_convertible` 进行检测

**元编程辅助**
- `is_type_condition<C>`: 识别类型特征模板（如 `std::is_reference`，拥有 `static constexpr bool value` 成员）
- `is_fundamental<T>`: 判断是否为基础类型（算术类型、void、nullptr_t），包装标准库版本以修复 MSVC bug

## TypeTraits.cpp 文件

仅包含头文件包含，无具体实现代码（纯头文件库）。

---

### 要点总结

- **目的**: 编译时类型特性检测和元编程工具库
- **技术**: SFINAE、`std::void_t`、模板特化
- **应用场景**: 泛型编程、容器适配、类型安全检查
- **主要检测项**:
  - 操作符支持（相等比较、哈希）
  - 函数/函数对象/Lambda 类型识别
  - 模板实例化检查
  - 基础类型判断
