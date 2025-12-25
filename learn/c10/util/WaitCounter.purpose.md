## WaitCounter 模块分析

**WaitCounter** 是一个用于监测和跟踪等待时间的计数器系统。它采用工厂模式和动态后端加载机制，允许多个不同的后端实现同时运行。

### 核心架构

**接口层** (`WaitCounter.h`)
- `WaitCounterBackendIf`: 后端接口，定义 `start()` 和 `stop()` 方法
- `WaitCounterBackendFactoryIf`: 工厂接口，用于创建具体的后端实例
- `WaitCounterHandle`: 用户面向的句柄类，提供简洁的 API
- `WaitGuard`: RAII 风格的守卫类，自动调用 `stop()`

**实现层** (`WaitCounter.cpp`)
- `DynamicBackendWrapper`: 包装动态加载的后端
- `WaitCounterImpl`: 核心实现类，管理多个后端的生命周期和状态
- `getDynamicBackend()`: 通过 `dlsym` 动态加载后端初始化函数

### 工作流程

1. 用户创建 `WaitCounterHandle(key)` - 通过 key 获取或创建 `WaitCounterImpl` 单例
2. 调用 `handle.start()` 返回 `WaitGuard` - 遍历所有注册的后端，调用各自的 `start()`
3. `WaitGuard` 析构时自动调用 `stop()` - 用当前时间和保存的上下文调用所有后端的 `stop()`

### 后端管理

- 静态全局 `waitCounterBackendFactories()` 管理已注册的工厂
- `registerWaitCounterBackend()` 允许动态注册新工厂
- `getDynamicBackend()` 在运行时通过 dlsym 加载外部后端实现

### 关键特性

- **多后端支持**: 单个计数器可同时使用多个测量后端
- **线程安全**: 使用 `Synchronized<>` 包装共享状态
- **RAII 模式**: `WaitGuard` 确保配对的 start/stop 调用
- **动态加载**: 支持在不重新编译的情况下添加新的测量后端

---

### 功能总结

- 提供灵活的等待时间监测框架
- 支持多个并行的后端实现（静态注册 + 动态加载）
- 通过宏 `STATIC_SCOPED_WAIT_COUNTER` 简化作用域内的计数
- 时间戳以微秒精度存储和传递
