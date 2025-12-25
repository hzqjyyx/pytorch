## c10/util/FunctionRef.h 主要功能

这是一个从 LLVM 的 `llvm::function_ref` 修改而来的类模板，提供了一个轻量级的、非所有权的可调用对象引用。

**核心设计**：
- `function_ref` 是一个类型擦除（type-erasing）的包装器，可以引用任何可调用对象（函数指针、lambda、仿函数等）
- 使用两个成员变量实现：
  - `callback`：一个函数指针，指向回调函数
  - `callable`：一个 `intptr_t`，存储原始可调用对象的地址

**实现机制**：
- 模板特化针对 `Ret(Params...)` 签名
- 通过 `callback_fn` 静态方法进行类型转换和调用
- SFINAE 条件确保只接受兼容的可调用对象

**关键特性**：
- 不拥有被引用的可调用对象（non-owning reference）
- 仅适合作为函数参数，不应长期存储
- 支持默认构造和 `nullptr` 初始化
- 重载 `operator()` 用于调用，`operator bool()` 用于检查有效性

**使用场景**：
- 作为函数参数接收任何可调用对象
- 避免模板膨胀和虚函数开销
- 提供比 `std::function` 更高效的解决方案

**主要优点**：
- 零运行时开销（相比虚函数）
- 类型安全的类型擦除
- 轻量级和高效
