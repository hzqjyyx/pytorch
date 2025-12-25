## CachingHostAllocator 主要功能

**CachingHostAllocator** 是 PyTorch CUDA 的主机内存（pinned memory）缓存分配器。它通过重用已释放的固定内存来优化性能，避免频繁调用 `cudaFreeHost` 导致的设备同步开销。

### 核心设计

**EventPool 事件池**（22-67行）：
- 为每个 CUDA 设备维护一个事件对象池
- 避免频繁调用成本较高的 `cudaEventCreate/Destroy`
- 序列化事件创建以防并发调用的性能问题
- 通过 `get()` 方法获取事件，销毁时自动归还到池中

**CUDACachingHostAllocatorImpl 核心实现**（71-254行）：
- 继承自 `CachingHostAllocatorImpl<CUDAStream, EventPool::Event>`
- 实现三种主机内存分配方式的抽象接口：
  1. **标准方式**：`cudaHostAlloc()` 直接分配固定内存
  2. **高性能方式**：`cudaHostRegister()` 预先分配普通内存后固定
  3. **并行预热**：使用线程池并行预热内存页以减少锁持时间

**allocWithCudaHostRegister 并行预热机制**（203-253行）：
- 使用 `std::malloc` 分配内存
- 线程池并行化：将内存分成多个段，多线程并行预热页面
- 每个线程设置其对应段的第一个字节，触发缺页中断
- 最后统一调用 `cudaHostRegister()` 一次性固定所有页

**统计与管理**：
- 记录 `cudaHostAlloc`、`cudaFreeHost` 等操作的耗时
- 使用互斥锁保护统计数据
- 支持清空缓存、重置统计信息

### 公共接口（CachingHostAllocator.h）

- `getCachingHostAllocator()`：获取全局分配器实例
- `CachingHostAllocator_recordEvent()`：记录流中的事件，防止内存重用前的竞态
- `CachingHostAllocator_emptyCache()`：释放所有缓存的固定内存
- `HostAlloc()`：便捷分配函数
- `CachingHostAllocator_getStats()`、`resetAccumulatedStats()`、`resetPeakStats()`：统计接口

### 关键特性

- **设备上下文管理**：优先使用主上下文分配，支持统一寻址，避免跨 NUMA 节点问题
- **流事件同步**：记录异步复制所在的流，确保内存重用时设备操作已完成
- **后台线程支持**：可配置后台线程进行异步清理工作
- **性能统计**：精确追踪分配/释放操作的耗时

---

### 核心功能概览

- **缓存重用**：避免 `cudaFreeHost` 同步开销
- **事件池化**：减少事件对象创建销毁成本
- **并行预热**：多线程加速内存页预热
- **流同步跟踪**：保证异步操作完成后才重用内存
- **可配置策略**：支持标准分配、主机注册、线程并行度等选项
- **性能监测**：统计分配释放耗时和峰值使用量
