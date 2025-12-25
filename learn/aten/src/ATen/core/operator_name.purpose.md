## OperatorName 类型系统

这两个文件定义了 PyTorch 中的操作符名称表示系统。

### operator_name.h - 核心定义

**OperatorName 结构体** (第 16-54 行)
- 存储操作符的基础名称（`name`）和重载标签（`overload_name`）
- 支持命名空间前缀，格式如 `namespace::operator_name`
- `getNamespace()` 方法：从名称中提取命名空间部分
- `setNamespaceIfNotSet()` 方法：在原地修改字符串，为未指定的名称添加命名空间前缀（通过字符串移位操作）

**OperatorNameView 结构体** (第 59-75 行)
- `OperatorName` 的非所有权视图版本，使用 `std::string_view`
- 所有方法都是 `constexpr`，支持编译期计算
- `parse()` 静态方法：解析 `"foo.overload"` 或 `"foo"` 格式的字符串

**比较和哈希** (第 77-97 行)
- 重载 `==` 和 `!=` 操作符
- 在 `std::hash` 特化中支持哈希表存储，通过 XOR 组合名称和重载名称的哈希值

### operator_name.cpp - 字符串转换

- `toString()` 函数：通过 `operator<<` 将 `OperatorName` 转换为字符串
- `operator<<()` 函数：格式化输出为 `name.overload_name`（如果有重载名称）或仅 `name`

### 核心特点

- **动态命名空间注入**：支持运行时为操作符名称添加命名空间前缀
- **两层设计**：拥有字符串的 `OperatorName` 和轻量级视图的 `OperatorNameView`，后者可用于编译期场景
- **字符串操作优化注记**：代码注释指出这些函数较慢，暗示未来可能改进内部数据结构

### 用途摘要

- 统一表示 PyTorch 操作符的标准格式
- 支持操作符的命名空间管理
- 提供可哈希的操作符标识，用于映射和集合
- 实现编译期操作符名称解析（通过 View 版本）
