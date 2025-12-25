## TorchDispatchModeTLS - PyTorch 分发模式线程本地存储

这个模块管理 PyTorch 中 TorchDispatchMode 的线程本地状态。TorchDispatchMode 是一种机制，允许 Python 代码拦截和自定义 PyTorch 操作的行为。

### 核心数据结构

**TorchDispatchModeTLS** 包含两个存储模式的容器：

1. **stack_** - 用户模式栈：存储用户推送的 TorchDispatchMode（如 ProxyTorchDispatchMode）
2. **infra_modes_** - 基础设施模式数组：存储特定类型的内置模式（FAKE、PROXY、FUNCTIONAL），每种类型最多一个

### 主要功能

**模式管理：**
- `push_non_infra_mode_onto_stack()` - 将用户模式推入栈
- `pop_stack()` - 弹出栈顶模式（优先弹出用户模式）
- `set_mode()` / `unset_mode()` / `get_mode()` - 管理基础设施模式

**栈查询：**
- `get_stack_at(idx)` - 获取指定位置的模式（逻辑栈包含基础设施模式+用户模式）
- `stack_len()` - 计算逻辑栈总长度
- `any_modes_set()` - 检查是否有任何模式处于活跃状态

**状态管理：**
- `get_state()` / `set_state()` - 保存/恢复整个分发模式状态
- `dispatch_mode_enabled()` - 检查分发模式是否启用

**工具函数：**
- `to_string()` - 将模式键转换为人类可读的字符串（ProxyTorchDispatchMode / FakeTensorMode）

### 关键机制

- 当推入或移除模式时，自动管理 `DispatchKey::Python` 和 `DispatchKey::PythonTLSSnapshot` 的包含状态
- 当所有模式都被移除时，禁用这些 DispatchKey；当有模式存在时启用它们
- 使用 `TORCH_CHECK` 验证不变量（如防止重复设置同一基础设施模式）

### 总结

- **目的**：管理 TorchDispatchMode 对象的生命周期和优先级
- **存储**：线程本地 (thread_local) 变量存储全局状态
- **模式类型**：FAKE（虚假张量）、PROXY（代理）、FUNCTIONAL（函数式）
- **栈操作**：支持嵌套/堆叠多个用户模式，基础设施模式为单例
- **自动管理**：push/pop 时自动更新相关的 DispatchKey 标志
