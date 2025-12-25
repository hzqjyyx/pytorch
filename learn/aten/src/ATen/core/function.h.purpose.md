# ATen/core/function.h 文件分析

这个文件定义了 PyTorch JIT 编译器中的 `Function` 基类，是 JIT 函数执行的核心抽象。

## 主要组件

**Function 类** (第 39-113 行)
- 纯虚拟基类，代表一个没有隐式 `self` 对象的图函数
- 包含 schema 信息和执行管理器

**核心虚拟方法**：
- `run(Stack& stack)` - 同步执行函数
- `runAsync()` - 异步执行函数，支持任务启动器
- `call()` - 两个重载版本，用于不同解释器（server/mobile）与 Code 对象交互

**Schema 和查询接口**：
- `getSchema()` - 获取函数签名信息
- `qualname()` / `name()` - 获取限定名和简单名
- `num_inputs()` - 获取输入参数数量
- `setSchema()` - 设置函数签名

**其他功能**：
- `operator()` - 函数调用操作符，自动检查和规范化输入
- `ensure_defined()` - 延迟初始化钩子
- `doc_string()` - 获取文档字符串
- `isGraphFunction()` - 检查是否为图函数

## 关键类型定义

- `Stack` - std::vector<IValue> 的别名，表示执行栈
- `Kwargs` - 关键字参数映射
- `TaskLauncher` - 异步任务启动函数类型

---

**主要功能概括**：
- 定义 JIT 函数的抽象接口
- 支持同步和异步执行
- 管理函数 schema 和元数据
- 提供多种解释器的调用机制
