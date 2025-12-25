## Copy-on-Write (COW) 内存管理

这两个文件实现了PyTorch中的写时复制机制，用于优化存储克隆和内存共享。

### 核心概念

**COWDeleterContext**: 一个引用计数上下文，用于追踪多少个存储共享同一块内存。当引用计数大于1时，修改操作必须先复制数据。

### 主要函数

**`lazy_clone_storage(StorageImpl& storage)`** (COW.cpp:51-111)
- 创建存储的COW克隆，同时将原始存储转换为COW形式
- 处理三种场景：
  1. 简单数据指针 → 包装为COW上下文
  2. 已是COW数据指针 → 直接增加引用计数
  3. 其他上下文类型 → 返回nullptr（不支持）
- 返回新的StorageImpl对象或nullptr

**`materialize_cow_storage(StorageImpl& storage)`** (COW.cpp:113-150)
- 急进地复制COW存储数据，将其转换为普通存储
- 检查是否为最后一个引用：
  - 是：直接使用原数据
  - 否：调用allocator->clone()复制数据
- 递减引用计数并释放旧数据指针

**`has_simple_data_ptr()`** (COW.cpp:35-45)
- 检查存储是否有简单数据指针（无特殊上下文）

**`is_cow_data_ptr()`** (COW.cpp:47-49)
- 检查数据指针是否使用COW删除器

**`make_data_ptr()` 和 `copy_data_ptr()`** (COW.cpp:19-31)
- 工具函数：包装和复制COW数据指针

### 关键设计

- **引用计数**：通过COWDeleterContext管理
- **线程安全**：COWDeleterContext内部包含互斥锁保护并发访问
- **并行禁止**：materialize_cow_storage在parallel_for循环内禁用
- **上下文转移**：使用move_context()转移所有权

### 主要功能点

- 减少克隆存储时的内存复制
- 实现延迟复制（写时复制）的语义
- 处理盲别名（blind aliases）的同步问题
- 支持存储的物化（从COW转为普通存储）
- 不支持非标准上下文的数据指针
