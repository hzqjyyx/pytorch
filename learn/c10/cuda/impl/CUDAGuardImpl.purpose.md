这些文件实现了PyTorch中CUDA设备的守卫接口（Guard Interface），用于管理CUDA设备状态、流和事件。

**CUDAGuardImpl.h** 定义了 `CUDAGuardImpl` 结构体，继承自 `DeviceGuardImplInterface`，包含以下核心功能：

**设备管理：**
- `exchangeDevice()` - 交换当前CUDA设备并返回旧设备
- `getDevice()` - 获取当前CUDA设备
- `setDevice()` - 设置当前CUDA设备
- `uncheckedGetDevice()` / `uncheckedSetDevice()` - 无异常检查的设备操作
- `deviceCount()` - 获取可用CUDA设备数量

**流管理：**
- `getStream()` - 获取指定设备的当前流
- `getDefaultStream()` - 获取默认流
- `getNewStream()` / `getStreamFromGlobalPool()` - 从流池获取新流
- `exchangeStream()` - 交换当前流
- `queryStream()` / `synchronizeStream()` - 查询和同步流

**事件管理：**
- `createEvent()` - 创建CUDA事件，支持PyTorch和CUDA事件标志映射
- `destroyEvent()` - 销毁事件
- `record()` - 在流上记录事件
- `block()` - 让流等待事件
- `queryEvent()` - 查询事件完成状态
- `synchronizeEvent()` - 同步事件

**其他功能：**
- `synchronizeDevice()` - 同步整个设备
- `recordDataPtrOnStream()` - 在流上记录数据指针（用于内存管理）
- `elapsedTime()` - 计算两个事件之间的时间差

**CUDAGuardImpl.cpp** 仅包含一行核心代码：
- `C10_REGISTER_GUARD_IMPL()` - 向PyTorch框架注册CUDA守卫实现

**关键设计特点：**
- 所有操作都与GPU追踪（GPUTrace）集成，用于性能分析和调试
- 事件操作前后进行设备保存/恢复，避免改变全局设备状态
- 使用 `C10_CUDA_CHECK` 和 `C10_CUDA_CHECK_WARN` 宏处理CUDA错误
- 支持无异常检查的操作变体（`noexcept`）

**概要：**
- 抽象CUDA设备、流、事件的底层操作
- 提供统一的接口与PyTorch上层代码交互
- 管理设备上下文和同步原语
- 集成追踪和调试功能
