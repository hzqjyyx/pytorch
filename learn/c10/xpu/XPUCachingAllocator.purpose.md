# XPU Caching Allocator 核心功能分析

## 整体架构

这是 PyTorch 为 Intel XPU 设备实现的内存缓存分配器，采用三层架构：

1. **Block** - 内存块的基本单位
2. **DeviceCachingAllocator** - 单个设备的内存管理器
3. **XPUAllocator** - 全局分配器，管理多个设备

## 内存池分层策略

### 两级缓存池
- **Small Pool**: 管理 ≤1MB 的小内存块
- **Large Pool**: 管理 >1MB 的大内存块

### 分配规则
```
请求大小           预分配块大小        所属池
≤1MB              2MB               Small Pool
1MB - 10MB        20MB              Large Pool  
≥10MB             向上取整到2MB倍数   Large Pool
```

### 对齐和粒度
- 设备内存对齐：512 字节
- 最小块大小：512 字节
- 所有请求向上取整到 512 字节的倍数

## 核心数据结构

### Block (c10/xpu/XPUCachingAllocator.cpp:43-81)
```cpp
struct Block {
  DeviceIndex device;           // 设备索引
  sycl::queue* queue;          // SYCL 队列指针
  stream_set stream_uses;      // 使用该块的所有流
  size_t size;                 // 块大小
  size_t requested_size;       // 用户实际请求的大小
  BlockPool* pool;             // 所属内存池
  void* ptr;                   // 实际内存地址
  bool allocated;              // 是否已分配给用户
  Block *prev, *next;          // 分割链表
  int event_count;             // 待完成的事件数
}
```

### BlockPool (c10/xpu/XPUCachingAllocator.cpp:37-41)
- 使用 `std::set<Block*, Comparison>` 按 (queue, size, ptr) 排序存储空闲块
- 支持快速查找满足大小要求的最小可用块

### 比较器逻辑 (c10/xpu/XPUCachingAllocator.cpp:83-93)
```
优先级: queue地址 > 块大小 > 内存地址
```

## 主要操作流程

### 内存分配 malloc() (c10/xpu/XPUCachingAllocator.cpp:408-452)

1. **预处理**
   - 加锁保护并发访问
   - 调用 `process_events()` 回收已完成事件的块
   - 将请求大小向上取整到 512 字节倍数

2. **查找策略**
   ```
   尝试从缓存池获取 → 失败则从设备分配新块 
   → 仍失败则释放所有缓存后重试 → 最终失败抛出 OOM
   ```

3. **块分割判断** `should_split()` (c10/xpu/XPUCachingAllocator.cpp:330-337)
   - Small Pool: 剩余 ≥512 字节则分割
   - Large Pool: 剩余 >1MB 则分割

4. **分割实现** `alloc_found_block()` (c10/xpu/XPUCachingAllocator.cpp:347-390)
   - 创建新块覆盖所需大小
   - 将剩余部分作为新块放回缓存池
   - 更新前后指针维护分割链表

### 内存释放 free() (c10/xpu/XPUCachingAllocator.cpp:454-468)

1. **标记释放**
   - 设置 `allocated = false`
   - 更新统计信息

2. **异步安全处理**
   - 如果 `stream_uses` 非空：调用 `insert_events()` 注册事件
   - 如果没有流依赖：直接调用 `free_block()`

### 块回收 free_block() (c10/xpu/XPUCachingAllocator.cpp:167-189)

1. **合并相邻空闲块**
   - 检查 `prev` 和 `next` 是否可合并
   - 更新块大小和指针关系
   - 删除被合并的块

2. **放回缓存池**
   - 从 `active_blocks` 移除
   - 插入对应的 `BlockPool`

### 事件管理

#### insert_events() (c10/xpu/XPUCachingAllocator.cpp:392-400)
- 为每个使用过该块的流提交栅栏（barrier）
- 创建 SYCL 事件并与块关联
- 递增 `event_count`

#### process_events() (c10/xpu/XPUCachingAllocator.cpp:191-215)
- 轮询所有事件的完成状态
- 完成的事件递减块的 `event_count`
- 当 `event_count` 归零时调用 `free_block()`

### 流记录 recordStream() (c10/xpu/XPUCachingAllocator.cpp:470-476)
- 记录块在不同流上的使用情况
- 防止过早释放仍在使用的内存
- 同一流的重复记录会被忽略

## 内存释放策略

### 缓存清空 emptyCache() (c10/xpu/XPUCachingAllocator.cpp:478-481)
调用 `release_cached_blocks()`：

1. **同步等待** (c10/xpu/XPUCachingAllocator.cpp:320-328)
   - `synchronize_and_free_events()`: 等待所有事件完成
   - `syncStreamsOnDevice()`: 设备级同步

2. **释放未分割块** (c10/xpu/XPUCachingAllocator.cpp:309-318)
   - 只释放 `prev == nullptr && next == nullptr` 的完整块
   - 分割的块需要等待整个链表都空闲

### 实际内存释放 release_block() (c10/xpu/XPUCachingAllocator.cpp:287-307)
- 调用 `sycl::free()` 归还内存给设备
- 更新 `reserved_bytes` 统计
- 从池中删除块对象

## 统计信息

### DeviceStats 追踪的指标
- **allocated_bytes**: 当前分配给用户的内存
- **reserved_bytes**: 从设备申请的总内存（包括缓存）
- **active_bytes**: 活跃块占用的内存
- **requested_bytes**: 用户实际请求的内存（未对齐）

### 统计类型
- `AGGREGATE`: 所有内存的聚合统计
- `SMALL_POOL`: 小内存池统计
- `LARGE_POOL`: 大内存池统计

每种统计类型都记录：current（当前值）、peak（峰值）、accumulated（累积值）

## 多设备管理

### XPUAllocator (c10/xpu/XPUCachingAllocator.cpp:515-668)
- 维护 `vector<DeviceCachingAllocator>` 为每个设备创建独立分配器
- 使用 `flat_hash_map<void*, Block*>` 快速查找指针对应的块
- 集成 PyTorch 的 GPU 内存追踪钩子

### 线程安全
- `DeviceCachingAllocator`: 使用 `std::recursive_mutex`
- `XPUAllocator`: 使用 `std::mutex` 保护 `allocated_blocks` 映射

## 公共 API (XPUCachingAllocator.h)

```cpp
Allocator* get()                          // 获取全局分配器
void init(DeviceIndex device_count)       // 初始化设备数量
void emptyCache()                         // 清空所有设备缓存
void* raw_alloc(size_t size)              // 原始分配
void raw_delete(void* ptr)                // 原始释放
void recordStream(DataPtr, XPUStream)     // 记录流使用
DeviceStats getDeviceStats(DeviceIndex)   // 获取统计信息
void resetPeakStats(DeviceIndex)          // 重置峰值统计
void resetAccumulatedStats(DeviceIndex)   // 重置累积统计
```

## 关键设计特点

1. **内存复用**: 缓存释放的块避免频繁系统调用
2. **块分割/合并**: 减少内存碎片
3. **异步安全**: 通过事件机制确保块在所有操作完成后才回收
4. **流感知**: 跟踪跨流的内存使用避免数据竞争
5. **分层池**: 针对不同大小请求优化分配策略
6. **OOM 详细信息**: 失败时提供设备容量、已分配、已保留等详细信息

---

**ROCm/Backward 相关内容**：
- 文件中未包含 ROCm 特定代码
- 无反向传播（backward）相关逻辑
- 这是纯内存管理实现，不涉及自动微分
