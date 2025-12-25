`IdWrapper.h` 是一个模板类，用于创建类型安全的 ID 包装器。主要功能：

**核心机制：**
- 提供 `IdWrapper<ConcreteType, UnderlyingType>` 模板基类
- 子类通过继承它来包装一个底层类型的 ID（如 `uint32_t`）
- 自动生成相等性比较和哈希函数

**使用方式：**
1. 定义一个继承 `IdWrapper` 的结构体，指定具体类型和底层类型
2. 在全局命名空间调用 `C10_DEFINE_HASH_FOR_IDWRAPPER(MyIdType)` 宏

**提供的操作：**
- `operator==`：比较两个 ID 是否相等
- `operator!=`：比较两个 ID 是否不相等
- `hash_value()`：返回 ID 的哈希值
- `underlyingId()`：获取底层 ID 值

**设计特点：**
- 构造函数和方法都是 `constexpr`，支持编译期计算
- 友元函数方式实现运算符重载
- 仅依赖底层类型的相等性和哈希能力
- 轻量级包装，无额外开销

**实际用途：**
- 创建类型安全的 ID 类（如 `BackendID`, `DispatchKeyID` 等）
- 防止不同语义的 ID 被错误地混用
- 支持在哈希容器（`unordered_map`, `unordered_set`）中使用
