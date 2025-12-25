**ThreadLocalPythonObjects 主要功能分析**

这是一个线程本地存储机制，用于在 C++ 中管理 Python 对象的引用。它提供了一个静态接口来存储和检索与线程绑定的 Python 对象。

核心设计：
- 使用 `thread_local` 关键字确保每个线程拥有独立的 `py_objects` 实例
- 内部使用 `std::unordered_map` 存储键值对，其中值是 `std::shared_ptr<SafePyObject>`
- `SafePyObject` 提供了对 Python 对象的安全封装

功能接口：

- **set(key, value)**: 将 Python 对象存储到线程本地字典中
- **get(key)**: 从线程本地字典获取 Python 对象，如果 key 不存在会触发 TORCH_CHECK 断言
- **contains(key)**: 检查指定的 key 是否存在
- **set_state(state)**: 替换整个线程本地状态（用于状态恢复或切换）
- **get_state()**: 获取当前线程的完整状态快照

使用场景：
- 在多线程环境中安全地管理 Python 对象引用
- 支持线程状态的保存和恢复（对于上下文切换或嵌套调用很有用）

关键特性：
- 线程隔离：每个线程有独立的对象存储，无需加锁
- 引用计数管理：通过 `shared_ptr` 自动处理对象生命周期
- 状态原子性：可以整体保存/恢复线程状态
