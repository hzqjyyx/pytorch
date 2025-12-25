## HPUHooksInterface 文件分析

**HPUHooksInterface.h** 定义了一个 HPU（Habana Processing Unit）加速器的接口类：

- 继承自 `AcceleratorHooksInterface`，是一个虚拟接口
- 提供了多个虚函数的默认实现，都返回 false 或抛出错误
- `hasHPU()` - 检查是否有 HPU 设备，默认返回 false
- `getDeviceFromPtr()` - 从内存指针获取设备信息，默认抛出错误
- `isPinnedPtr()` - 检查内存是否被锁定，默认返回 false
- `getPinnedMemoryAllocator()` - 获取锁定内存分配器，默认抛出错误
- `hasPrimaryContext()` - 检查设备是否有主上下文，默认抛出错误
- 定义了 `HPUHooksRegistry` 注册表和 `REGISTER_HPU_HOOKS` 宏用于注册具体实现

**HPUHooksInterface.cpp** 实现了获取 HPU hooks 的工厂函数：

- `getHPUHooks()` - 返回全局单例 HPU hooks 对象
- 首先尝试从注册表创建具体实现（如果已注册 HPU 后端）
- 如果注册表中没有实现，则返回默认的 `HPUHooksInterface` 实例
- 使用静态变量保证单例模式

**核心设计：**

- 提供了一个扩展点，当 HPU 后端可用时可以注册具体实现
- 默认实现都会抛出错误，迫使用户在使用 HPU 功能前先注册后端
- 遵循注册表模式，支持运行时动态加载不同的 HPU 实现
