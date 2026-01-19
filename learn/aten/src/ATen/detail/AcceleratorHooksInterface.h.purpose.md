# AcceleratorHooksInterface.h 文件分析

这个文件定义了一个抽象接口 `AcceleratorHooksInterface`，用于所有加速器后端（如 CUDA、ROCm 等）提供统一的接口规范。

## 核心设计

- **目的**：提供一个共享接口，允许 CPU 代码以通用方式调用加速器功能
- **模式**：基于虚函数的 Hook 机制，各个后端实现具体逻辑

## 主要功能模块

**后端状态检查**
- `isBuilt()`：检查后端是否在编译时启用
- `isAvailable()`：检查后端是否在运行时可用（已构建、驱动可用、有支持的设备）
- `hasPrimaryContext()`：检查设备是否已初始化

**设备管理**
- `init()`：初始化后端
- `deviceCount()`：获取设备数量
- `setCurrentDevice()`：设置当前活跃设备
- `getCurrentDevice()`：获取当前活跃设备
- `exchangeDevice()`：交换设备并返回前一个设备
- `maybeExchangeDevice()`：可选的设备交换

**内存管理**
- `isPinnedPtr()`：检查指针是否指向固定内存
- `getPinnedMemoryAllocator()`：获取固定内存分配器
- `getDeviceFromPtr()`：从指针获取对应设备

**随机数生成**
- `getDefaultGenerator()`：获取默认生成器
- `getNewGenerator()`：创建新生成器

## 关键特性

- 所有方法都是虚函数，默认实现返回 false 或抛出错误
- 使用 `TORCH_CHECK` 确保未实现的功能会明确报错
- `isBuilt()` 和 `isAvailable()` 保证不抛异常
- 采用 `[[maybe_unused]]` 标记可能未使用的参数
