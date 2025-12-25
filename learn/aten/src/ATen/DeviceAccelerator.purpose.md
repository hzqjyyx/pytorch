**DeviceAccelerator 模块的主要功能：**

该模块定义了 PyTorch 的**顶级加速器设备抽象**，用于统一管理 CUDA、HIP、XPU、MPS、HPU、MTIA 和 PrivateUse1 等互斥的加速器设备。

**核心设计原则：**
- 同一二进制文件中只能有一个加速器可用
- 所有加速器都支持异步计算和 Stream/Event 系统
- 提供统一的设备管理接口

**主要 API 功能：**

- **`getAccelerator(bool checked)`** - 检测并返回当前可用的加速器设备类型
  - 优先级：PrivateUse1 > 运行时检测（MTIA）> 编译时检测（CUDA/XPU/HIP/MPS/HPU）
  - 若 checked=true，保证返回有效值；checked=false 时可返回 nullopt

- **`isAccelerator(DeviceType)`** - 判断设备类型是否为加速器

- **`isAcceleratorExcluded(DeviceType, ...excluded)`** - 判断设备是否为加速器且不在排除列表中

- **`deviceCount()`** - 返回当前加速器的设备数量（不抛异常）

- **`setDeviceIndex(index)` / `getDeviceIndex()`** - 设置/获取当前活跃设备

- **`setCurrentStream(stream)` / `getCurrentStream(index)`** - 设置/获取设备流

- **`synchronizeDevice(index)`** - 同步指定设备上的所有待处理操作

**实现细节：**

- 使用宏定义 `DETECT_RUNTIME_ACCELERATOR` 和 `DETECT_AND_ASSIGN_ACCELERATOR_COMP` 实现设备检测
- 通过 `c10::impl::VirtualGuardImpl` 实现对不同设备类型的统一操作
- 设有校验机制防止多个加速器同时被编译进同一二进制文件
