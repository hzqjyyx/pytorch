## Python GIL 死锁检测机制

这两个文件实现了 PyTorch 中用于检测 Python GIL（全局解释器锁）相关死锁的工具。

### 核心设计

**问题背景**：在多线程或需要阻塞操作的场景中，如果持有 Python GIL 会导致死锁。这个模块提供了一种机制来在执行可能阻塞的操作前断言确保 GIL 未被持有。

**架构特点**：
- 采用虚拟调用的间接方式，避免在非 Python 依赖的代码中直接引入 Python 头文件
- 通过注册器模式动态注入 GIL 检查的具体实现

### 主要组件

**DeadlockDetection.h**：
- `TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP()` 宏：用于在非 Python 上下文中断言 GIL 未被持有
- `PythonGILHooks` 虚拟接口：定义 GIL 检查的抽象行为
- `PythonGILHooksRegisterer` 注册器：在对象生命周期内注册/注销 GIL 检查实现
- `SetPythonGILHooks()` 和 `check_python_gil()` 函数：管理和查询 GIL 状态

**DeadlockDetection.cpp**：
- 全局变量 `python_gil_hooks` 指针存储注册的 GIL 检查实现
- `disable_detection()` 函数：通过环境变量 `TORCH_DISABLE_DEADLOCK_DETECTION` 允许禁用检测
- `check_python_gil()` 函数：调用注册的虚拟实现检查 GIL（无注册时返回 false）
- `SetPythonGILHooks()` 函数：管理注册（若检测被禁用则直接返回）

### 关键特性

- **环境变量控制**：支持通过 `TORCH_DISABLE_DEADLOCK_DETECTION` 禁用检测
- **torchdeploy 兼容性**：在 torchdeploy 环境中始终报告 GIL 未被持有
- **断言失败信息**：提供帮助链接 (issue #56297) 指导用户如何正确释放 GIL

### 功能要点

- 通过虚拟接口支持多个 Python 实现（CPython、torchdeploy 等）的 GIL 检查
- 使用 RAII 模式（注册器析构函数）自动清理 GIL 检查实现
- 防止在 torchdeploy 实例中重复注册导致的覆盖问题
- 在没有 Python 链接的模块中仍能进行 GIL 检查（通过间接调用）
