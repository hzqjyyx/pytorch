## 主要功能

这两个文件提供了 PyTorch 中 CUDA 设备管理的核心 C++ API 封装。主要功能包括：

### 1. 设备计数与初始化

**`device_count()` (c10/cuda/CUDAFunctions.cpp:97-114)**
- 静态初始化，只执行一次设备计数
- 容错设计：即使 CUDA 初始化失败也不抛异常，返回 0
- 处理多种错误场景：无设备、驱动版本不足、初始化错误等

**`device_count_impl()` (c10/cuda/CUDAFunctions.cpp:17-94)**
- 实际执行设备计数的内部函数
- 详细的错误处理：
  - `cudaErrorNoDevice`: 返回 0
  - `cudaErrorInsufficientDriver`: 检查驱动版本，给出具体升级建议
  - `cudaErrorInitializationError`: GPU 不存在
  - `cudaErrorUnknown`: 环境配置问题（如修改 `CUDA_VISIBLE_DEVICES`）
  - ASAN 模式下的内存分配错误特殊处理

### 2. 设备切换与上下文管理

**CUDA 12+ 的特殊处理 (c10/cuda/CUDAFunctions.cpp:212-295)**

CUDA 12 引入了关键行为变化：`cudaSetDevice` 会立即创建 primary context，而非延迟创建。为避免分布式训练中不必要的显存占用，引入了新的机制：

**`SetDevice()` (c10/cuda/CUDAFunctions.cpp:233-242)**
- 清除 `targetDeviceIndex`
- 检查当前设备，避免重复设置
- 调用 `cudaSetDevice`

**`MaybeSetDevice()` (c10/cuda/CUDAFunctions.cpp:244-250)**
- 核心创新：只有当目标设备已有 primary context 时才调用 `cudaSetDevice`
- 否则只保存设备索引到 `targetDeviceIndex`，避免创建上下文
- 用于 device guard 析构函数和 `torch.cuda.device` 上下文管理器

**`ExchangeDevice()` vs `MaybeExchangeDevice()` (c10/cuda/CUDAFunctions.cpp:254-289)**
- `ExchangeDevice`: 总是初始化目标设备的 CUDA 上下文
- `MaybeExchangeDevice`: 只在目标设备已有上下文时才切换，否则仅设置 `targetDeviceIndex`

**`SetTargetDevice()` (c10/cuda/CUDAFunctions.cpp:291-295)**
- 将延迟的设备切换实际执行
- 如果 `targetDeviceIndex >= 0`，调用 `SetDevice` 完成真正的切换

**向后兼容示例**：
```python
x = torch.empty(1, device="cuda:1")  # cuda:0 上不创建上下文
y = torch.empty(1, device="cuda")    # 此时才在 cuda:0 创建上下文
```

### 3. 同步操作

**`device_synchronize()` (c10/cuda/CUDAFunctions.cpp:137-144)**
- GPU 追踪钩子集成
- 等待计数器（性能分析）
- 调用 `cudaDeviceSynchronize()`

**`warn_or_error_on_sync()` (c10/cuda/CUDAFunctions.cpp:148-154)**
- 调试模式支持：`SyncDebugMode::L_ERROR` 抛异常，`L_WARN` 发出警告
- 帮助用户发现意外的同步操作

**内联同步函数 (c10/cuda/CUDAFunctions.h:77-111)**
- `memcpy_and_sync()`: 异步拷贝 + 流同步（HIP 3.1+ 使用 `hipMemcpyWithStream`）
- `stream_synchronize()`: 流同步
- 都包含调试模式检查和 GPU 追踪

### 4. Primary Context 查询

**`hasPrimaryContext()` (c10/cuda/CUDAFunctions.cpp:186-188)**
- 通过函数指针间接调用，实现延迟绑定
- 由 `CUDAHooks.cpp` 在运行时设置实际实现

**`getDeviceIndexWithPrimaryContext()` (c10/cuda/CUDAFunctions.cpp:156-172)**
- 查找已有 primary context 的设备
- 优先检查当前设备，再遍历所有设备
- 返回 `std::optional<DeviceIndex>`

### 5. CUDA < 12 的简化实现

对于 CUDA 11 及以下版本，延迟上下文创建是默认行为：
- `MaybeSetDevice()` 直接等同于 `SetDevice()`
- `MaybeExchangeDevice()` 直接等同于 `ExchangeDevice()`
- `SetTargetDevice()` 是空操作
- 无需 `targetDeviceIndex` 机制

---

## ROCm/HIP 相关
- `hipMemcpyWithStream` (HIP 3.1+) 替代 `cudaMemcpyAsync` + 同步

## 向后兼容处理
- CUDA 12+ 通过 `targetDeviceIndex` 机制避免过早创建上下文
- 保持与旧版 CUDA 的行为一致性
