# Symbol.h 文件分析

这个文件定义了 PyTorch 中的 **Symbol** 类，用于在 IR（中间表示）中表示和管理各种操作符和属性的符号。

## 核心概念

Symbol 是一种特殊的"interned string"（内联字符串），带有命名空间结构。它采用 `uint32_t` 作为内部唯一标识符，支持高效的命名空间测试和哈希操作。

## 主要符号类型

文件注释中提到三大类符号：

1. **prim** - 原始符号，仅存在于 IR 中的合成操作符
2. **onnx** - ONNX 标准操作符，语义由 ONNX 规范定义
3. **attr** - 属性键，ONNX 和 ATen 操作符共享

## 核心设计策略

- 使用枚举生成连续的整数序列作为唯一标识
- 通过 constexpr Symbol 将枚举值转换为实际的 Symbol 类型
- 支持在 switch 语句中使用（通过 `operator unique_t()` 转换）

## 主要功能

- **创建符号**：从限定字符串（"attr::bar"）或域+非限定字符串创建
- **命名空间支持**：attr、aten、cuda、onnx、prim、user、caffe2、dimname、scope
- **类型检查**：is_attr()、is_aten()、is_prim() 等判断方法
- **字符串转换**：
  - `toUnqualString()` - 无限定名称（"mm"）
  - `toQualString()` - 限定名称（"aten::mm"）
  - `domainString()` - 域名称（"org.pytorch.aten"）
- **哈希支持**：std::hash 特化，支持在哈希表中使用

## 要点总结

- Symbol 是 uint32_t 包装器，提供命名空间化的操作符/属性标识
- 支持多种符号域（aten、onnx、prim、attr 等）
- 高效的命名空间测试和哈希操作
- 支持字符串与符号之间的双向转换
- 设计用于 IR 表示和编译器优化
