## CUDAContext.h 文件

这是一个头文件，主要作用是：
- 包含 `CUDAContextLight.h`（轻量级 CUDA 上下文定义）
- 为了向后兼容性，保留了对其他头文件的包含（Context.h、CUDAStream.h、Logging.h、Exceptions.h）

## CUDAContext.cpp 文件

实现了 CUDA 设备上下文管理的核心功能：

**静态全局变量（匿名命名空间）：**
- `num_gpus`: 记录系统中 GPU 设备数量
- `device_flags`: 用于一次性初始化每个设备的同步机制
- `device_properties`: 存储所有 GPU 的设备属性

**初始化函数：**
- `initCUDAContextVectors()`: 初始化上述全局向量，使用静态初始化器确保只执行一次
- `initDeviceProperty()`: 获取指定设备的 CUDA 属性（通过 `cudaGetDeviceProperties`）

**公开 API 函数：**
- `warp_size()`: 返回当前设备的 warp 大小
- `getCurrentDeviceProperties()`: 获取当前设备的属性指针
- `getDeviceProperties()`: 获取指定设备（或当前设备）的属性指针，使用 `call_once` 确保属性只初始化一次
- `canDeviceAccessPeer()`: 检查一个设备是否能够访问另一个设备的内存（peer-to-peer 访问）
- `getCUDADeviceAllocator()`: 返回 CUDA 内存分配器

## 总结

- **设备属性管理**: 缓存并延迟加载各 GPU 设备的属性信息
- **线程安全**: 使用 `call_once` 机制确保初始化只发生一次
- **设备验证**: 包含边界检查确保设备索引有效
- **P2P 支持**: 提供设备间通信能力检查
- **内存分配**: 提供 CUDA 内存分配器访问接口
