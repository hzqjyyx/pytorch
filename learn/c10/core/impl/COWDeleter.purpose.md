## Copy-on-Write (COW) 内存管理实现

这组文件实现了一个 Copy-on-Write 机制，用于优化 PyTorch 中的数据指针管理，避免不必要的数据复制。

### 核心组件

**COWDeleterContext 类** (`COWDeleter.h:16-56`)
- 持有原始数据指针和对应的删除器函数
- 通过引用计数追踪有多少个 DataPtr 共享这份数据
- 使用 `std::shared_mutex` 保护数据访问，允许多个读者但只有一个写者

**引用计数管理**
- `increment_refcount()` (`COWDeleter.cpp:18-21`): 增加引用计数
- `decrement_refcount()` (`COWDeleter.cpp:23-36`): 减少引用计数，返回 `std::variant`：
  - 如果不是最后一个引用，返回 `std::shared_lock` 让调用者在读锁下复制数据
  - 如果是最后一个引用，返回 `std::unique_ptr` 包装的原始数据，之后删除上下文对象本身

**cow_deleter 函数** (`COWDeleter.cpp:7-9`)
- 作为 DataPtr 的删除器回调
- 将 void 指针转换为 COWDeleterContext，触发引用计数递减和可能的销毁

### 关键设计

- 引用计数初始化为 1 (`COWDeleter.h:55`)
- 当最后一个引用被删除时，COWDeleterContext 自我销毁 (`COWDeleter.cpp:31`)
- 防止嵌套的 COW 包装 (`COWDeleter.cpp:15`)
- 使用 `shared_mutex` 实现读写分离的锁定策略

### 总结

- **目的**: 实现零复制的数据共享机制
- **机制**: 引用计数 + 共享互斥锁
- **触发**: 当多个 DataPtr 指向同一块数据时
- **优势**: 延迟数据复制直到必要时刻（真正需要修改时）
- **自动清理**: 最后一个引用移除时自动销毁上下文和原始数据
