# CUDACachingAllocator 核心功能

## 主要目的
实现 CUDA 设备内存的缓存分配器，避免频繁调用 `cudaMalloc/cudaFree`，通过内存池复用来提高性能。

## 核心架构

### 内存组织层次
- **Block**: 内存分配的基本单元（可能是某次 cudaMalloc 的一部分）
  - 包含设备、流、大小、指针、分配状态等信息
  - 通过 `prev/next` 指针形成双向链表，表示从同一段连续内存分割出的块
  - 记录 `stream_uses` 用于多流同步
  
- **BlockPool**: 管理未分配的缓存块
  - `small_blocks`: ≤1MB 的小块（打包在 2MB 缓冲区中）
  - `large_blocks`: >1MB 的大块（独立分配或使用 20MB 缓冲区）
  - 使用有序集合按大小或地址排序，快速查找合适的块

- **PrivatePool**: CUDA Graph 专用的私有内存池
  - 图捕获时从独立池分配，避免重放时地址被其他操作占用
  - 维护独立的 `large_blocks` 和 `small_blocks`

### 分配策略

**尺寸划分**:
- `kMinBlockSize = 512B`: 最小对齐单位
- `kSmallSize = 1MB`: 小/大块分界线
- `kSmallBuffer = 2MB`: 小块打包缓冲区
- `kLargeBuffer = 20MB`: 中等大小（1-10MB）的分配缓冲区
- `kRoundLarge = 2MB`: 大块对齐单位

**分配流程** (`malloc`):
1. 尝试从现有池中找到最小可用块（best-fit）
2. 找不到则调用 `cudaMalloc` 申请新内存
3. OOM 时逐步释放缓存：
   - 释放足够大的未分割块
   - 释放所有未分割块
   - 执行垃圾回收（如启用）

**释放流程** (`free`):
- 标记 Block 为未分配
- 如果有跨流使用（`stream_uses`），插入 CUDA Event 延迟释放
- 否则立即归还到 BlockPool

### 可扩展段（Expandable Segments）

**动态内存映射机制**（仅 Linux + Driver API）:
- 使用 `cuMemAddressReserve` 预留巨大虚拟地址空间（~256TiB）
- 按需通过 `cuMemCreate` + `cuMemMap` 增加物理内存映射
- 粒度：小池 2MB/大池 20MB 的段
- 优势：减少批次大小变化导致的碎片化
- 限制：不支持 IPC（需使用 pidfd 系统调用在进程间共享）

**实现要点**:
- `ExpandableSegment` 管理虚拟地址范围和物理页句柄
- `map(range)` 映射新物理内存
- `unmap(range)` 解除映射归还页面
- 支持多设备 peer access（手动设置访问权限）

### CUDA Graph 集成

**捕获期间的内存隔离**:
- `beginAllocateToPool` 注册过滤器，匹配的流分配走私有池
- `captures_underway` 非空时跳过 `process_events`（避免非法查询）
- `needs_events_deferred_until_no_capture` 延迟事件记录

**检查点与恢复** (`setCheckpointPoolState`):
- 图录制后保存私有池状态（`PrivatePoolState`）
- 重放后需录制新图时，恢复到检查点状态
- 处理"录制间活跃张量"：释放新增分配，重建检查点分配

### 事件同步

**EventPool**: 
- 每设备维护 CUDA Event 对象池（避免频繁创建/销毁）
- 线程安全的获取/归还机制

**跨流使用追踪**:
- `recordStream` 记录块在其他流上的使用
- `insert_events` 为每个使用流插入事件
- `process_events` 查询事件完成状态，完成后真正释放块

### 统计与监控

**DeviceStats 指标**:
- 分配/保留/活跃字节数（按 AGGREGATE/SMALL_POOL/LARGE_POOL 统计）
- 分段数、分割块数、超大分配数
- OOM 次数、重试次数、设备同步次数

**历史追踪** (`recordHistory`):
- `RingBuffer<TraceEntry>` 记录分配/释放/OOM 事件
- 可选记录调用栈（通过 `CreateContextFn`）
- 三种级别：STATE（仅活跃分配）/ALLOC（含历史）/ALL（含释放）

### IPC 支持

**共享句柄** (`shareIpcHandle`):
- 普通分配：导出 `cudaIpcMemHandle_t`
- 可扩展段：导出文件描述符（需 `pidfd_getfd` 系统调用）
- 返回偏移量 + 序列化句柄

**接收方** (`getIpcDevPtr`):
- 解析句柄类型（CUDA_MALLOC / EXPANDABLE_SEGMENT）
- 恢复内存映射或导入 IPC 句柄

### 内存碎片管理

**减少碎片的策略**:
- `max_split_size`: 超过此阈值的块不分割（减少碎片）
- `garbage_collection_threshold`: 内存压力下触发 GC
- 分割块统计：`inactive_split`/`inactive_split_bytes`
- 可扩展段优先填充低地址空隙（促进合并）

### 关键常量与配置

```cpp
kMinBlockSize = 512          // 最小对齐
kSmallSize = 1048576         // 1 MB
kSmallBuffer = 2097152       // 2 MB
kLargeBuffer = 20971520      // 20 MB
kMinLargeAlloc = 10485760    // 10 MB
kRoundLarge = 2097152        // 2 MB
```

### 接口层次

**CUDAAllocator 抽象类**:
- `raw_alloc/raw_delete`: 原始分配接口
- `recordStream`: 跨流使用注册
- `emptyCache`: 清空缓存
- `snapshot`: 导出完整内存状态
- `getCheckpointState/setCheckpointPoolState`: 检查点功能

**全局访问**:
- `c10::cuda::CUDACachingAllocator::allocator` 原子指针
- 提供 `get()->method()` 形式的包装函数

---

## 简要列举（忽略内容）

**ROCm 相关**:
- `#ifdef USE_ROCM` 条件编译分支
- HIPify 转换支持

**Backward Compatibility**:
- 保留 `DeviceStats` 类型别名
- `FreeCudaMemoryCallbacksRegistry` 旧回调机制
- `THCCachingAllocator_` 命名空间演进说明
