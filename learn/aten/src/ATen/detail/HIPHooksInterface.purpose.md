## HIPHooksInterface 功能分析

HIPHooksInterface 是 PyTorch ATen 中用于动态分派 HIP（AMD GPU 计算框架）功能的接口系统。

**设计模式：**
- 继承自 `AcceleratorHooksInterface`，采用虚基类模式
- 注册表模式：通过 `C10_DEFINE_REGISTRY` 和 `TORCH_DECLARE_REGISTRY` 实现动态注册和查询
- 延迟加载：`getHIPHooks()` 使用静态变量缓存，避免重复初始化

**关键机制：**
- `.h` 文件定义接口规范，包含所有虚函数声明
- `.cpp` 文件实现 `getHIPHooks()` 工厂函数
- 在非移动端环境下，优先从注册表中创建实际的 HIP 实现
- 若注册表中无实现，降级到基础的 `HIPHooksInterface` 空实现（所有操作都报错）

**虚函数职责：**
- `hasHIP()`：检查是否有可用的 HIP 支持
- `current_device()`：获取当前 GPU 设备索引
- `getDefaultGenerator()`：获取默认随机数生成器
- `getPinnedMemoryAllocator()`：获取固定内存分配器
- `getNumGPUs()`：获取 GPU 数量
- `hasPrimaryContext()`：检查设备主上下文

**关键特征：**

- CPU 代码通过此接口调用 HIP 功能，避免直接依赖 HIP 库
- 支持编译时可选的 HIP 支持（通过 `C10_MOBILE` 宏控制）
- 所有基类实现均返回错误提示，迫使实际 HIP 实现必须覆盖相关方法
- 线程安全的单例模式存储 hooks 实例

**主要目的：**

- 实现 HIP 库的松耦合集成
- 支持在无 HIP 支持的环境下编译和运行 CPU 代码
- 通过注册表机制允许外部 HIP 实现库动态注入自己的功能
