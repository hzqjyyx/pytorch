## PrivateUse1HooksInterface 功能分析

这个接口是 PyTorch 为第三方加速器（如 NPU、MLU 等）提供的**插件化集成机制**。

### 核心设计

**PrivateUse1HooksInterface.h** 定义了一个虚拟接口类，继承自 `AcceleratorHooksInterface`，提供了加速器需要实现的标准方法：

- `getDefaultGenerator()` - 获取默认随机数生成器
- `getNewGenerator()` - 创建新的随机数生成器
- `getDeviceFromPtr()` - 从内存指针获取设备信息
- `isPinnedPtr()` - 检查是否为固定内存
- `getPinnedMemoryAllocator()` - 获取固定内存分配器
- `hasPrimaryContext()` - 检查设备主上下文是否存在
- `resizePrivateUse1Bytes()` - 调整设备存储大小

所有未实现的方法都通过 `FAIL_PRIVATEUSE1HOOKS_FUNC` 宏抛出 `NOT_IMPLEMENTED` 异常。

**PrivateUse1HooksInterface.cpp** 提供了**单例注册和获取机制**：

- 使用 `std::mutex` 保护全局 `privateuse1_hooks` 指针
- `RegisterPrivateUse1HooksInterface()` - 线程安全地注册实现（仅允许注册一次）
- `isPrivateUse1HooksRegistered()` - 检查是否已注册
- `detail::getPrivateUse1Hooks()` - 获取已注册的实现（未注册时抛出异常）

### 工作流程

第三方加速器在初始化时创建 `PrivateUse1HooksInterface` 的实现类并调用 `RegisterPrivateUse1HooksInterface()` 注册，PyTorch 核心代码后续通过 `detail::getPrivateUse1Hooks()` 调用相应的加速器功能。

### 关键特性

- **延迟绑定** - 运行时动态注册加速器实现
- **线程安全** - 使用互斥锁保护注册过程
- **单例模式** - 全局仅允许一个加速器实现
- **错误处理** - 未实现或未注册时明确的异常提示

### 要点总结

- 提供第三方加速器的标准接口定义
- 支持运行时插件化集成机制
- 通过线程安全的单例注册和获取实现
- 包含随机数生成、内存管理、设备查询等核心功能接口
- 未实现的方法主动抛异常，提示第三方需要实现
