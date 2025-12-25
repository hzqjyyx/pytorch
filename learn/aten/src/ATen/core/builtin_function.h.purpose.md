# BuiltinOpFunction 类

`BuiltinOpFunction` 是一个继承自 `Function` 的结构体，用于表示内置操作函数。

## 核心组成

**构造函数** (lines 13-23)
- 接收函数的限定名 (`QualifiedName`)、函数签名 (`FunctionSchema`)、可调用对象和文档字符串
- 断言函数返回值数量为 1
- 存储这些信息作为成员变量

**执行接口**
- `run()` (lines 29-31)：直接执行存储的可调用对象
- `runAsync()` (lines 33-40)：同步执行后，将结果包装成 `Future` 对象返回

**元数据接口**
- `doc_string()` (lines 25-27)：返回函数文档字符串
- `qualname()` (lines 42-44)：返回函数的限定名称
- `getSchema()` (lines 51-53)：返回函数签名
- `num_inputs()` (lines 55-57)：返回参数数量

**其他方法**
- `setSchema()` (lines 59-62)：修改函数签名
- `call()` 重载 (lines 64-76)：兼容不同的调用接口，最终都调用 `run()`
- `ensure_defined()` (lines 47-49)：空实现（此类函数已定义）

## 主要功能

- **包装 C++ 函数**：将 std::function 包装成 JIT 可执行的函数对象
- **提供函数元数据**：管理函数名称、签名、文档等信息
- **支持异步执行**：提供 Future 基础的异步调用机制
- **统一 JIT 接口**：实现 Function 基类的所有虚方法

## 关键特性

- 私有成员变量存储函数名、可调用对象、签名和文档
- 支持同步和异步两种执行方式
- 作为 Torch JIT 系统中的基础函数类型
