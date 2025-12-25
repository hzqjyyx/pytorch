**HIPHooksInterface 概览**

这是 PyTorch 中用于 HIP（AMD ROCm 对应的异构计算接口）的动态分派机制。由于 HIP 代码需要单独编译，CPU 代码无法直接调用 HIP 功能，因此使用注册表模式来实现运行时动态加载。

**头文件 (HIPHooksInterface.h)**

- 定义 `HIPHooksInterface` 结构体，继承自 `AcceleratorHooksInterface`
- 提供纯虚接口方法的默认实现，当 HIP 库未加载时抛出错误：
  - `init()` - 初始化 HIP
  - `getDefaultGenerator()` - 获取默认随机数生成器
  - `hasHIP()` - 检查 HIP 是否可用
  - `current_device()` - 获取当前设备索引
  - `isPinnedPtr()` - 检查是否为固定内存指针
  - `getPinnedMemoryAllocator()` - 获取固定内存分配器
  - `getNumGPUs()` - 获取 GPU 数量
  - `hasPrimaryContext()` - 检查是否有主上下文
- 定义 `HIPHooksRegistry` 注册表，用于注册实际的 HIP 实现
- 提供 `REGISTER_HIP_HOOKS` 宏便于子类注册

**源文件 (HIPHooksInterface.cpp)**

- 实现 `getHIPHooks()` 函数，使用懒初始化模式：
  - 非移动端：尝试从注册表创建 HIP 实现
  - 如果注册表中无实现，返回默认的空实现
  - 使用 `static` 变量保证全局单例
- 定义 `HIPHooksRegistry` 的实际注册表对象

**核心设计特点**

- **动态分派**：不硬依赖 HIP 库，支持可选编译
- **惰性加载**：首次调用时才初始化 hooks
- **优雅降级**：无 HIP 库时提供默认实现，返回错误或空值
- **注册表模式**：允许不同的 HIP 实现在编译时注册

**主要职能**

- 作为 CPU 代码与 HIP 计算库之间的桥梁
- 管理 HIP 设备状态和内存分配
- 提供随机数生成器等计算基础设施的统一接口
