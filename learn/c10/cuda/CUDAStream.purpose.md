# CUDAStream 核心功能

## 1. 流池管理系统

实现了一个高效的 CUDA 流池化方案，避免频繁创建/销毁流的开销：

**池结构 (c10/cuda/CUDAStream.cpp:44-54)**
```cpp
static std::array<
    std::array<
        std::array<cudaStream_t, kStreamsPerPool>,  // 32个流
        C10_COMPILE_TIME_MAX_GPUS>,                  // 每个设备
    c10::cuda::max_compile_time_stream_priorities>   // 每个优先级
    streams;
```

- 每个设备有多个优先级池（默认最多4个）
- 每个优先级池预分配32个流 (`kStreamsPerPool = 1 << 5`)
- 采用 round-robin 轮询分配策略 (c10/cuda/CUDAStream.cpp:252-255)

## 2. StreamId 编码机制

使用 64 位整数巧妙编码流信息 (c10/cuda/CUDAStream.cpp:69-96)：

```
 [54 bits: zeros] [5 bits: index] [4 bits: type] [1 bit: native/ext]
```

**三种流类型：**
- **DEFAULT (0x0)**: 默认流，StreamId = 0
- **PRIORITY (1-N)**: 原生优先级流，最低位为1
- **EXT (0xF)**: 外部流，直接存储 `cudaStream_t` 指针，最低位为0

**编码/解码函数：**
- `makeStreamId()` (c10/cuda/CUDAStream.cpp:159-165): 构造 StreamId
- `streamIdType()` (c10/cuda/CUDAStream.cpp:140-152): 提取流类型
- `streamIdIndex()` (c10/cuda/CUDAStream.cpp:154-157): 提取池索引

## 3. 线程安全的当前流管理

**线程本地存储 (c10/cuda/CUDAStream.cpp:169)**
```cpp
static thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;
```

每个线程维护独立的"当前流"数组（每个设备一个），但流池是全局共享的。这意味着：
- 不同线程可以在同一个流上同步
- 流本身是线程安全的（CUDA 保证）
- 当前流设置是线程隔离的

## 4. 延迟初始化策略

采用两级延迟初始化：

**全局初始化 (c10/cuda/CUDAStream.cpp:173-198)：**
- 通过 `c10::call_once` 保证只执行一次
- 查询设备数量和优先级范围
- 计算实际支持的优先级数量

**设备级初始化 (c10/cuda/CUDAStream.cpp:217-226)：**
- CUDA: 首次请求设备流时创建整个池
- ROCm: 首次使用时才创建单个流（懒加载）

## 5. 核心 API 实现

**从池获取流 (c10/cuda/CUDAStream.cpp:315-332)：**
```cpp
CUDAStream getStreamFromPool(const int priority, DeviceIndex device_index)
```
- 将优先级映射到池索引：`pri_idx = clamp(-priority, 0, max-1)`
- 原子递增计数器实现 round-robin
- 返回包装后的 `CUDAStream` 对象

**流转换 (c10/cuda/CUDAStream.cpp:269-310)：**
`cudaStream_t CUDAStream::stream() const` 将 StreamId 解码为实际的 `cudaStream_t`：
- DEFAULT → `nullptr`（CUDA 默认流约定）
- EXT → 直接 reinterpret_cast 指针
- PRIORITY → 从 `streams[]` 数组取出预创建的流

**当前流操作：**
- `getCurrentCUDAStream()` (c10/cuda/CUDAStream.cpp:357-365): 返回线程当前流
- `setCurrentCUDAStream()` (c10/cuda/CUDAStream.cpp:367-370): 设置线程当前流

## 6. CUDAStream 类设计 (c10/cuda/CUDAStream.h)

**核心职责：**
- 封装 `c10::Stream` 并强制类型为 CUDA
- 提供到 `cudaStream_t` 的隐式转换
- 实现同步、查询、优先级操作

**关键方法：**
- `query()` (c10/cuda/CUDAStream.h:115-129): 非阻塞检查流是否完成
- `synchronize()` (c10/cuda/CUDAStream.h:131-134): 阻塞等待流完成
- `priority()` (c10/cuda/CUDAStream.h:136-141): 查询流优先级
- `pack3()/unpack3()` (c10/cuda/CUDAStream.h:160-170): 序列化支持

## 7. 资源管理特性

**流泄漏策略 (c10/cuda/CUDAStream.cpp:35-39)：**
预创建的流永不销毁，避免 CUDA runtime 已析构时调用 `cudaStreamDestroy` 导致崩溃。

**GPU Trace 集成 (c10/cuda/CUDAStream.cpp:207-212)：**
创建流时通知 Python 解释器，支持性能分析工具。

---

## ROCm/HIP 差异
- HIP 流较重，采用单流懒加载而非整池预创建
- HIP 优先级范围不同：1=低，0=默认，-1=高（vs CUDA 的 0=默认，负数=高）
- 需要额外的 `stream_flags` 数组实现懒加载同步
