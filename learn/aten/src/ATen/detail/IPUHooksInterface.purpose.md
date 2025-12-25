## IPUHooksInterface 功能分析

**IPUHooksInterface.h** 定义了 IPU（Graphcore Intelligence Processing Unit）加速器的钩子接口：

- `IPUHooksInterface` 结构体继承自 `AcceleratorHooksInterface`，是 IPU 设备的抽象接口
- 实现了虚函数：`init()`、`hasPrimaryContext()`、`getDefaultGenerator()`、`getNewGenerator()`
- 所有虚函数都抛出 `TORCH_CHECK` 错误，提示"Cannot initialize IPU without ATen_ipu library"
- 定义了 `IPUHooksArgs` 空结构体，用于注册参数
- 使用宏 `REGISTER_IPU_HOOKS` 便于派生类注册到 IPU 钩子注册表

**IPUHooksInterface.cpp** 实现了钩子的获取机制：

- `getIPUHooks()` 函数返回全局单例的 `IPUHooksInterface` 实例
- 使用注册表 `IPUHooksRegistry()->Create()` 尝试创建具体的 IPU 实现
- 如果注册表中有实现类，使用该实现；否则创建默认的空实现
- 通过 `C10_DEFINE_REGISTRY` 宏定义 IPU 钩子的全局注册表

**核心设计模式：**

- **Registry Pattern（注册表模式）**：允许动态注册 IPU 实现
- **Bridge Pattern（桥接模式）**：通过接口将 PyTorch 核心与 IPU 库解耦
- **Lazy Initialization（延迟初始化）**：首次调用 `getIPUHooks()` 时才创建实例
- 当 ATen_ipu 库不可用时，提供优雅降级处理（默认实现抛出错误提示）

**主要功能点：**

- IPU 设备初始化接口
- IPU 设备上下文管理
- IPU 随机数生成器管理
- 可插拔的 IPU 实现注册机制
