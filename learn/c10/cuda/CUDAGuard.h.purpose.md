## CUDAGuard.h 文件功能

该文件定义了四个 RAII 风格的 CUDA 设备和流管理类：

**CUDAGuard**
- 在构造时设置指定的 CUDA 设备，析构时恢复原设备
- 接受 `DeviceIndex` 或 `Device` 对象初始化
- 提供 `set_device()`、`reset_device()`、`set_index()` 方法动态修改设备
- 可查询原始设备和当前设备

**OptionalCUDAGuard**
- CUDAGuard 的可选版本，可以不初始化（uninitialized state）
- 接受 `std::optional<Device>` 或 `std::optional<DeviceIndex>`
- 支持 `reset()` 方法恢复未初始化状态

**CUDAStreamGuard**
- 管理 CUDA 流和关联的设备
- 构造时设置指定流（同时设置该流所属的设备），析构时恢复
- 提供 `reset_stream()` 方法重置为新流
- 可查询原始/当前流和设备

**OptionalCUDAStreamGuard**
- StreamGuard 的可选版本
- 支持延迟初始化，可接受 `std::optional<Stream>`
- 提供 `reset()` 方法返回未初始化状态

**CUDAMultiStreamGuard**
- 管理多个 CUDA 流（在不同设备上）
- 接受 `ArrayRef<CUDAStream>`

---

**核心特性：**
- 所有类禁用复制和移动构造/赋值（RAII 安全）
- 内部使用 `InlineDeviceGuard` 和 `InlineStreamGuard` 模板实现
- 提供类型安全的设备/流管理，避免直接调用 CUDA API
