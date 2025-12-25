## LegacyVmapMode 功能分析

**LegacyVmapMode** 是 PyTorch 中用于跟踪和管理 `torch.vmap`（向量化映射）嵌套层级的机制。

### 核心设计

采用**线程本地存储**（thread-local storage）维护一个全局计数器 `VmapMode_current_vmap_level`，用来记录当前线程进入了多少层嵌套的 vmap。

### 主要操作

1. **current_vmap_level()**
   - 返回当前的 vmap 嵌套层级
   - 每进入一个 vmap 就会增加，离开就会减少

2. **increment_nesting()**
   - vmap 进入时调用，计数器加 1
   - 当计数器从 0 变为 1 时，启用 `DispatchKey::VmapMode` 在所有张量上
   - 这样让 PyTorch 的调度系统知道当前正在执行 vmap 操作

3. **decrement_nesting()**
   - vmap 退出时调用，计数器减 1
   - 当计数器从 1 变为 0 时，禁用 `DispatchKey::VmapMode`
   - 恢复正常的张量操作行为

### 与调度系统的集成

通过 `c10::impl::tls_set_dispatch_key_included()` 与 PyTorch 的动态调度系统（Dispatcher）关联，使得 vmap 相关的算子实现能被正确路由到对应的 vmap 处理逻辑。

---

**关键点：**
- 线程安全的计数器跟踪 vmap 嵌套深度
- 在计数转换时触发调度系统的启用/禁用
- 仅用于 Python `torch.vmap` API 的实现支持
- 不是 C++ API（C++ vmap API 尚不存在）
