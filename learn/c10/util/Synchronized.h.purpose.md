## Synchronized.h 文件分析

这是一个线程安全的数据包装类，提供简单的互斥锁保护机制。

**核心设计：**

- 包装任意类型 `T` 的数据，使用 `std::mutex` 保护访问
- 灵感来自 Facebook 的 Folly 库，但实现了精简版本
- 仅提供 `withLock()` 方法作为唯一的 API

**主要特性：**

- 两个构造函数：默认构造和值/右值引用初始化
- 禁用拷贝和移动操作（因为 `std::mutex` 不可复制/移动）
- 两个重载的 `withLock()` 方法：
  - 非 const 版本：接收可修改的数据引用
  - const 版本：接收 const 引用的数据
- 都返回回调函数的返回值

**使用模式：**

```cpp
Synchronized<int> counter(0);
counter.withLock([](int& value) {
    value++;
});
```

**Key Points：**

- 自动化的互斥锁管理（RAII 模式）
- 防止数据竞态（data race）
- 轻量级实现，仅覆盖最常用的同步需求
- 不支持更复杂的 Folly Synchronized 功能（如 upgrade locks）
