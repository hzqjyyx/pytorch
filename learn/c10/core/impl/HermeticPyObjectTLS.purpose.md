## HermeticPyObjectTLS - PyObject 隔离控制机制

这个模块实现了一个**线程本地存储（TLS）机制**，用于控制 PyObject 与 Tensor 的关联行为。

### 核心概念

**Hermetic Mode（隔离模式）**：当启用时，系统不会将 PyObject 永久关联到 Tensor。每次访问相同的 Tensor 代码时，都会获得一个新的 PyObject 实例，而不是重用之前缓存的对象。

### 实现细节

**HermeticPyObjectTLS.h**（头文件）：
- 定义了 `HermeticPyObjectTLS` 结构体，提供静态接口
- `set_state(bool)`：设置隔离模式状态
- `get_state()`：获取当前状态（包含快速路径优化的注释，但当前被禁用）
- `init_state()`：初始化状态（供 torchdeploy/multipy 调用）
- 使用 `std::atomic<bool>` 保证线程安全
- 关键设计：状态只从 `false` 转换到 `true`，不会反向转换

**HermeticPyObjectTLS.cpp**（实现文件）：
- 定义 `thread_local std::atomic<bool> hermeticPyObjectState`：每个线程独立的状态标志
- 定义 `static std::atomic<bool> haveState_`：全局原子标志，标记是否初始化过
- `set_state()`：更新线程本地的隔离状态
- `get_tls_state()`：读取线程本地状态
- `init_state()`：设置全局初始化标志

### 同步设计

采用弛豫内存序（`memory_order_relaxed`）的原子操作，因为：
- 状态只会单向转换（false → true）
- 多解释器场景下有额外的同步保证
- 避免了不必要的内存屏障开销

---

### 功能总结

- **用途**：控制 PyTorch 中 Python 对象与张量的绑定策略
- **应用场景**：多解释器环境（torchdeploy/multipy）
- **线程安全**：使用 thread_local 和原子操作
- **内存序**：弛豫原子操作，适应多解释器同步模型
- **状态转换**：单向转换（false → true），支持惰性初始化
