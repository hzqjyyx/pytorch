这个文件实现了基于 CUDA 11.4+ 的 `cudaMallocAsync` API 的内存分配器，作为 PyTorch 传统缓存分配器的替代方案。

## 核心架构

**PtrUsage 结构** (c10/cuda/CUDAMallocAsyncAllocator.cpp:55-63)
- 跟踪每个已分配指针的元数据：
  - `creation_stream`: 原始分配流
  - `recorded_streams`: 通过 `record_stream()` 记录的副使用流
  - `size`: 分配大小
  - `captured`: 标记是否在 CUDA graph capture 期间分配

**全局状态管理** (c10/cuda/CUDAMallocAsyncAllocator.cpp:65-102)
- `ptr_info`: 扁平哈希映射，存储所有活跃指针的元数据
- `pytorch_used_bytes` / `pytorch_memory_limits`: 按设备限制 PyTorch 内存使用（与同进程其他库共享 mempool 时的隔离）
- `dummy_unifying_free_streams`: 每设备一个虚拟流，用于统一多流释放

## 关键机制

### 1. 多流安全释放 (c10/cuda/CUDAMallocAsyncAllocator.cpp:192-271)

**单流场景**：直接在创建流上调用 `cudaFreeAsync`

**多流场景** (c10/cuda/CUDAMallocAsyncAllocator.cpp:214-265)：
```
张量在流 A 创建，在流 B、C 上被使用
问题：cudaFreeAsync 只接受单个"最近使用流"
解决：
  1. 使用虚拟"统一流"
  2. 同步统一流与所有使用流（创建流 + 所有 recorded_streams）
  3. 在统一流上执行 cudaFreeAsync
```

这确保释放操作等待所有使用流完成，避免 use-after-free。

### 2. CUDA Graph Capture 兼容性

**问题 1：释放未捕获的分配** (c10/cuda/CUDAMallocAsyncAllocator.cpp:79-92)
- Capture 期间禁止释放非 captured 分配（CUDA 限制）
- 检测到此情况时延迟释放：加入 `ungraphed_ptrs_defer_free_until_no_capture`
- 在 capture 结束后或下次非 capture 分配时批量释放 (c10/cuda/CUDAMallocAsyncAllocator.cpp:335-345)

**问题 2：悬空释放流** (c10/cuda/CUDAMallocAsyncAllocator.cpp:107-140)
- Capture 图必须满足：所有分支流最终汇合到初始流
- 张量析构时机不可控，可能在流汇合后才调用，导致图中出现悬空 free 节点
- 解决：capture 期间记录所有释放流到 `capture_free_streams`
- `endAllocateToPool` 时手动同步所有释放流到 capture 流 (c10/cuda/CUDAMallocAsyncAllocator.cpp:819-830)

### 3. 惰性设备初始化 (c10/cuda/CUDAMallocAsyncAllocator.cpp:147-180)

首次使用设备时：
- 获取默认 mempool 并配置：
  - `ReleaseThreshold = UINT64_MAX`: 永不主动释放给系统（最大程度复用）
  - 启用事件依赖复用、机会性复用、内部依赖复用
- 创建虚拟统一流（从流池获取）
- 初始化内存限制为无限

### 4. 内存限制实现 (c10/cuda/CUDAMallocAsyncAllocator.cpp:356-390)

不依赖 CUDA 的软 `ReleaseThreshold`（会引入性能非确定性），而是：
- 分配前检查：`pytorch_used_bytes[device] + size > pytorch_memory_limits[device]`
- 超限直接返回 `cudaErrorMemoryAllocation`
- 成功后更新 `pytorch_used_bytes` 计数

### 5. Workspace 估算 (c10/cuda/CUDAMallocAsyncAllocator.cpp:522-595)

为 cuDNN 的 `cudnnFind` 提供最大可用 workspace：
- 计算理论上限：`min(device_free, limit - used)`
- 二分试探：尝试分配，成功则返回，失败则减半重试
- 不需要精确，依赖 cuDNN 缓存机制逐步收敛到稳定值

## CudaMallocAsyncAllocator 类实现

**核心分配接口**：
- `allocate()` / `raw_alloc()`: 调用内部 `mallocAsync()` (c10/cuda/CUDAMallocAsyncAllocator.cpp:317-401)
- `raw_delete()`: 调用内部 `freeAsync()` (c10/cuda/CUDAMallocAsyncAllocator.cpp:273-313)

**流管理**：
- `recordStream()`: 将流添加到 `recorded_streams` 集合 (c10/cuda/CUDAMallocAsyncAllocator.cpp:610-631)
- 与创建流相同时发出警告（无效操作）

**统计信息** (c10/cuda/CUDAMallocAsyncAllocator.cpp:692-753)：
- 直接查询 CUDA mempool 属性：
  - `cudaMemPoolAttrUsedMemCurrent/High` → allocated/active bytes
  - `cudaMemPoolAttrReservedMemCurrent/High` → reserved bytes
- 不区分 allocated 和 active（native allocator 的概念）

**Graph 生命周期钩子**：
- `beginAllocateToPool()`: 设置 `capture_underway = true` (c10/cuda/CUDAMallocAsyncAllocator.cpp:792-803)
- `endAllocateToPool()`: 同步所有 free 流并清理状态 (c10/cuda/CUDAMallocAsyncAllocator.cpp:805-838)

**P2P 支持** (c10/cuda/CUDAMallocAsyncAllocator.cpp:879-894)：
- `cudaMallocAsync` 的 mempool 需要显式启用跨设备访问
- 使用 `cudaMemPoolSetAccess()` 而非 `cudaDeviceEnablePeerAccess()`

## 不支持的功能

文件中多个接口抛出错误，要求用户提交 issue：
- IPC 句柄共享 (`shareIpcHandle`, `getIpcDevPtr`)
- 分配历史记录 (`recordHistory`)
- OOM 观察器 (`attachOutOfMemoryObserver`)
- Checkpoint/恢复 (`getCheckpointState`, `setCheckpointPoolState`)
- 快照 (`snapshot` - 因为不跟踪独立块)

---

**ROCm/Backward 相关**：
- 文件开头 `#if CUDA_VERSION >= 11040` 保护整个实现
- 低版本 CUDA 返回 nullptr 并报错 (c10/cuda/CUDAMallocAsyncAllocator.cpp:928-931)
- 无 ROCm 特定代码（ROCm 尚未支持异步分配 API）
