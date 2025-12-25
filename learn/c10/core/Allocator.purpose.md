# c10/core/Allocator 核心功能解析

## 1. 内存管理抽象层

这两个文件定义了 PyTorch 的核心内存分配器接口，为不同设备（CPU、CUDA等）提供统一的内存管理抽象。

### DataPtr - 智能指针封装
- 封装了 `void*` 指针 + 自定义删除器 + 设备信息
- 支持存储额外的上下文（context）用于删除时传递信息
- 允许在运行时比较和替换删除器（`compare_exchange_deleter`）
- 零大小分配也保留设备信息，确保处理一致性

## 2. Allocator 接口设计

### 核心虚函数
```cpp
virtual DataPtr allocate(size_t n) = 0;              // 分配内存
virtual void copy_data(void* dest, const void* src, size_t count) const = 0;  // 数据拷贝
```

### 提供的功能
- **clone()** (Allocator.cpp:10-14): 分配新内存并拷贝数据
- **raw_allocate/raw_deallocate** (Allocator.h:190-199): 为 Thrust 等库提供原始指针接口
- **default_copy_data()** (Allocator.cpp:16-21): 使用 `std::memcpy` 的默认实现

### 设计考量
- `raw_deleter()` 返回非空时，表示支持原始指针接口（数据指针 == 上下文指针）
- `is_simple_data_ptr()` 检查是否为简单数据指针（无特殊上下文）

## 3. 全局分配器注册机制

```cpp
static std::array<Allocator*, COMPILE_TIME_MAX_DEVICE_TYPES> allocator_array{};
static std::array<uint8_t, COMPILE_TIME_MAX_DEVICE_TYPES> allocator_priority{};
```

- **SetAllocator()** (Allocator.cpp:47-52): 按优先级设置设备分配器
- **GetAllocator()** (Allocator.cpp:54-58): 获取设备对应的分配器
- **REGISTER_ALLOCATOR 宏** (Allocator.h:282-285): 静态注册分配器
- 优先级机制允许覆盖默认分配器（更高优先级才能覆盖）

## 4. 性能分析集成

通过 `ThreadLocalDebugInfo` 集成内存分析：

- **memoryProfilingEnabled()** (Allocator.cpp:60-64): 检查是否启用分析
- **reportMemoryUsageToProfiler()** (Allocator.cpp:66-78): 报告内存使用
- **reportOutOfMemoryToProfiler()** (Allocator.cpp:80-91): 报告 OOM 事件

`MemoryReportingInfoBase` 接口允许 profiler 收集：
- 分配大小、总分配量、总预留量
- 设备信息、OOM 事件

## 5. InefficientStdFunctionContext

为用户友好接口提供任意 `std::function` 删除器支持：

- 双重动态分配（context 本身 + std::function）
- **makeDataPtr()** (Allocator.cpp:31-40): 创建带自定义删除器的 DataPtr
- 实现了移动语义但禁止拷贝
- 析构时自动调用用户提供的删除函数

## 6. CachingAllocator 统计工具

提供内存池统计支持（Allocator.h:334-412）：

### Stat 结构
- `current`: 当前使用量
- `peak`: 峰值使用量  
- `allocated`/`freed`: 累计分配/释放量
- `increase()`/`decrease()`: 更新统计并维护峰值

### StatType 枚举
- `AGGREGATE`: 聚合统计
- `SMALL_POOL`: 小对象池
- `LARGE_POOL`: 大对象池

### DurationStat 结构
- 记录操作耗时的 `total`/`count`/`min`/`max`
- 用于分析分配器性能

---

**ROCm/Backward 相关**:
- 无 ROCm 特定代码
- 无反向传播相关逻辑
- 纯前向基础设施组件
