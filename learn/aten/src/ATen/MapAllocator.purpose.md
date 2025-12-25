## MapAllocator 核心功能

提供跨平台的内存映射文件分配器，支持进程间共享内存。主要用于在 PyTorch 中实现高效的大规模数据共享和持久化。

### 主要类

**1. MapAllocator**

基础内存映射分配器，封装了文件映射的底层操作：

- **平台适配**：通过条件编译处理 Windows (`CreateFileMapping`/`MapViewOfFile`) 和 POSIX (`mmap`) 的差异
- **文件操作模式**（通过 flags 控制）：
  - `ALLOCATOR_MAPPED_SHARED`: 共享文件映射（多进程可见）
  - `ALLOCATOR_MAPPED_SHAREDMEM`: 使用 `shm_open` 的共享内存对象
  - `ALLOCATOR_MAPPED_EXCLUSIVE`: 独占创建（O_EXCL）
  - `ALLOCATOR_MAPPED_NOCREATE`: 仅打开已存在的映射
  - `ALLOCATOR_MAPPED_KEEPFD`: 保持文件描述符打开
  - `ALLOCATOR_MAPPED_FROMFD`: 从已有文件描述符创建
  - `ALLOCATOR_MAPPED_UNLINK`: 映射后删除文件

**2. RefcountedMapAllocator**

带引用计数的共享内存分配器（继承自 MapAllocator）：

- **原子引用计数**：在映射内存的头部存储 `std::atomic<int>` 引用计数
- **自动清理**：最后一个引用释放时自动 unlink 共享内存
- **对齐处理**：预留 64 字节（`map_alloc_alignment`）存储元数据，`data()` 返回偏移后的指针
- **平台限制**：仅在支持原子操作的平台启用（`AT_ATOMIC_IPC_REFCOUNT`）

### 关键实现细节

**构造流程**（MapAllocator::MapAllocator）

1. **参数验证**：检查 flags 组合合法性
2. **文件/共享内存创建或打开**：
   - Windows: `CreateFileW`/`CreateFileMappingW` 或 `shm_open` 模拟
   - POSIX: `open`/`shm_open` + `fstat`
3. **大小调整**：如果请求大小大于文件，使用 `ftruncate`（POSIX）或 `SetEndOfFile`（Windows）扩展
4. **内存映射**：
   - Windows: `MapViewOfFile`
   - POSIX: `mmap` with `MAP_SHARED` 或 `MAP_PRIVATE`
5. **优化提示**：Linux 上调用 `posix_fadvise(POSIX_FADV_SEQUENTIAL)` 提升 CUDA 上传速度
6. **资源管理**：根据 flags 决定是否关闭文件描述符或 unlink 文件

**Windows 特殊处理**

- 使用 Event 对象（`CreateEventW`）同步多进程访问
- `WaitForReleaseHandle` 回调：在进程退出时异步清理 handle
- UTF-16 路径转换（`c10::u8u16`）

**共享内存命名**（NewProcessWideShmHandle）

生成唯一名称：`/torch_{pid}_{random}_{counter}`

### DataPtr 集成

通过静态工厂方法返回 `at::DataPtr`：

```cpp
at::DataPtr MapAllocator::makeDataPtr(filename, flags, size, actual_size_out)
```

- 内部创建 `new MapAllocator`，将其作为 context 存储在 DataPtr 中
- 删除器函数 `deleteMapAllocator` 负责析构
- 支持 `fromDataPtr` 反向提取 allocator

### 典型使用场景

1. **进程间张量共享**：PyTorch 多进程数据加载器通过共享内存传递张量
2. **大文件映射**：直接将磁盘文件映射为张量，避免全量加载
3. **持久化存储**：将张量数据持久化到文件系统

---

**其他内容（简要）**：
- ROCm 相关：无（此文件无 ROCm 特定代码）
- Backward 相关：无（纯内存管理，不涉及自动微分）
- 平台兼容性：完整支持 Windows/Linux/macOS，Android 部分功能受限（无 `posix_fadvise`）
- 错误处理：所有失败路径使用 `TORCH_CHECK` 抛出异常，包含详细的 errno/GetLastError 信息
