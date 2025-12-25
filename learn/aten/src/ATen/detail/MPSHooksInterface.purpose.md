## MPSHooksInterface 文件分析

**MPSHooksInterface.h** 定义了一个接口类，用于抽象 Metal Performance Shaders (MPS) 后端的功能。这是 PyTorch 为 Apple 硅芯片 GPU 加速设计的接口。

接口包含以下主要虚函数：

- **初始化与检测**：`init()`、`hasMPS()`、`isOnMacOSorNewer()` - 用于初始化 MPS 库和检测硬件支持
- **内存管理**：`getMPSDeviceAllocator()`、`emptyCache()`、`getCurrentAllocatedMemory()`、`getDriverAllocatedMemory()`、`getRecommendedMaxMemory()`、`setMemoryFraction()` - 控制 GPU 内存分配和监控
- **随机数生成器**：`getDefaultGenerator()`、`getNewGenerator()` - 管理 MPS 设备的随机数生成器
- **设备同步**：`deviceSynchronize()`、`commitStream()` - 确保 GPU 操作完成
- **性能分析**：`profilerStartTrace()`、`profilerStopTrace()` - 用于性能追踪
- **事件管理**：`acquireEvent()`、`releaseEvent()`、`recordEvent()`、`waitForEvent()`、`synchronizeEvent()`、`queryEvent()`、`elapsedTimeOfEvents()` - GPU 事件同步和计时
- **缓冲区访问**：`getCommandBuffer()`、`getDispatchQueue()` - 获取底层 Metal API 对象
- **固定内存**：`isPinnedPtr()`、`getPinnedMemoryAllocator()` - 处理 CPU-GPU 共享内存
- **上下文检测**：`hasPrimaryContext()` - 检查设备主上下文

**MPSHooksInterface.cpp** 实现了 `getMPSHooks()` 工厂函数，通过注册表机制动态创建 MPS 实现或返回默认的存根实现。

### 核心特点：
- **注册表模式**：使用 `C10_DEFINE_REGISTRY` 支持动态加载 MPS 实现
- **存根实现**：所有虚函数默认抛出 TORCH_CHECK 异常，指示 MPS 后端不可用
- **条件编译**：非移动平台上优先加载已注册的 MPS 实现
- **单例模式**：使用静态变量缓存 hooks 对象

### 主要功能列表：

- MPS 后端抽象接口定义
- GPU 内存管理（分配、缓存、监控）
- 随机数生成器管理
- 设备同步与流管理
- 性能分析与事件追踪
- Metal API 底层对象访问
- 动态实现加载（通过注册表）
- 存根实现（MPS 不可用时报错）
- 单例 hooks 对象管理
