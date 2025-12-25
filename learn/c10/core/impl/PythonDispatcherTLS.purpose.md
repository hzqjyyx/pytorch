## PythonDispatcherTLS 功能分析

这个模块实现了一个**线程本地存储(TLS)机制**来管理 Python 派发器的状态。

### 核心设计

**PythonDispatcherTLS** 是一个静态接口，通过线程本地变量 `pythonDispatcherState` 存储当前线程的 `PyInterpreter*` 指针：

- **set_state()**: 设置状态并同时更新 DispatchKey 的包含标志。当设置非空状态时，将 `DispatchKey::PythonDispatcher` 标记为已启用；当设置空指针时调用 `reset_state()`
- **get_state()**: 返回当前线程的 Python 解释器指针
- **reset_state()**: 清空状态指针并禁用 `DispatchKey::PythonDispatcher`

**DisablePythonDispatcher** 是一个 RAII 风格的作用域守卫，用于临时禁用 Python 派发器：

- 构造时：保存当前状态并清空派发器
- 析构时：自动恢复之前的状态
- 删除了所有复制和移动操作，确保不可转移和不可复制

### 关键特性

- **线程安全**: 每个线程独立维护自己的派发器状态
- **作用域管理**: DisablePythonDispatcher 通过 RAII 保证状态恢复
- **调度集成**: 与 c10 的 DispatchKey 系统集成，动态控制派发键的启用状态

### 主要功能

- 管理 Python 派发器在线程级别的启用/禁用状态
- 提供作用域内的临时禁用机制，确保异常安全
- 与 PyTorch 的多派发系统协调，控制何时进行 Python 层面的操作派发
