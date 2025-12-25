## SafePyObject 文件分析

SafePyObject 是 PyTorch 的 C10 核心库中用于安全管理 Python 对象引用的组件。

**SafePyObject（所有权持有者）：**
- 提供了一个类似 pybind11 的 `py::object` 的安全包装器，用于持有 PyObject 的所有权
- 核心特点是多解释器安全（multi-interpreter safe），支持 torchdeploy 等多解释器场景
- 在获取底层 PyObject* 指针时，必须指定当前的解释器上下文，并进行验证匹配
- 支持移动语义（move constructor）但禁用了移动赋值操作符
- 实现了完整的复制构造和赋值操作符，会自动管理 Python 对象的引用计数（incref/decref）
- 析构时自动递减引用计数

**SafePyObjectT（类型安全的泛型包装）：**
- 在 SafePyObject 基础上增加类型标签 `T`，用于类型安全检查
- 禁用复制和移动赋值操作，更加严格的所有权管理
- `T` 仅用作类型标签，不实际存储数据

**SafePyHandle（非所有权引用）：**
- 与 SafePyObject 相反，不持有 PyObject 的所有权
- 用于引用全局 Python 对象，这些对象会在解释器退出时泄漏
- 支持复制构造和赋值，更灵活的引用方式
- 提供 `reset()` 方法清空引用，`operator bool()` 检查有效性

**主要功能总结：**

- SafePyObject：所有权持有的安全 Python 对象包装器，多解释器安全
- SafePyObjectT：带类型标签的 SafePyObject，增强类型安全性
- SafePyHandle：非所有权的 Python 对象引用，用于全局对象
- 解释器上下文验证：ptr() 方法检查调用者的解释器与对象的解释器匹配
- 自动引用计数管理：构造、复制、赋值、析构时自动处理
- 禁用 Tensor 对象存储：文档明确指出不应用此类存储 Tensor，应直接使用 TensorImpl
