## Storage.h

定义了 `Storage` 类，是 PyTorch 中内存管理的核心抽象。主要功能：

- **存储初始化**：支持多种构造方式
  - 通过 `StorageImpl` 指针创建
  - 使用分配器动态分配内存
  - 使用预分配的 `DataPtr` 创建

- **内存数据访问**：
  - `data()`/`mutable_data()` - 获取原始指针
  - `data_ptr()` - 获取 `DataPtr` 对象（带删除器）
  - `set_data_ptr()` - 替换底层数据指针

- **存储属性**：
  - `nbytes()` / `sym_nbytes()` - 获取字节大小
  - `resizable()` - 判断是否可重新分配
  - `device_type()` / `device()` - 获取设备信息
  - `allocator()` - 获取分配器

- **引用计数管理**：
  - `use_count()` / `unique()` - 检查引用计数
  - `is_alias_of()` - 判断是否共享同一内存

- **特殊用途**：
  - Legacy API 支持（`create_legacy()`, `reset_legacy()`）
  - 外部指针共享（`UniqueStorageShareExternalPointer()`）
  - 借用语义支持（`unsafe_borrow_t`, `MaybeOwnedTraits`）

## Storage.cpp

只有一个核心函数 `isSharedStorageAlias()`：

- **功能**：判断两个 `Storage` 对象是否指向同一块内存
- **实现**：
  - 检查两个存储的删除器是否都是 `refcounted_deleter`
  - 比较删除器上下文（context）是否相同
  - 上下文相同意味着指向同一个引用计数对象

---

**核心功能总结：**
- • 内存生命周期管理（分配、释放、重新分配）
- • 跨设备存储支持（CPU/CUDA）
- • 引用计数与别名检测
- • DataPtr 包装（内存 + 删除器 + 上下文）
- • 支持外部内存管理集成
