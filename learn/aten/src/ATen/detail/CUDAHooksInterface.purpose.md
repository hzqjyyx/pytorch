## CUDAHooksInterface 文件分析

这两个文件实现了一个**动态分发机制**，允许 CPU 代码调用 CUDA 功能，同时保持编译时的解耦。

### 核心设计原理

PyTorch 将代码分为两个共享库：
- **libATen_cpu.so** - CPU 库（主库）
- **libATen_cuda.so** - CUDA 库（可选，动态加载）

CUDAHooksInterface 是一个**虚基类**，定义了 CPU 代码可能需要调用的所有 CUDA 功能的接口。

### 工作流程

1. **接口定义** (CUDAHooksInterface.h)
   - 定义纯虚函数，提供 CUDA 功能的默认实现（抛出错误）
   - 这些默认实现告诉用户需要加载 CUDA 库

2. **单例管理** (CUDAHooksInterface.cpp)
   - `getCUDAHooks()` 函数负责创建/返回 CUDAHooksInterface 实例
   - 使用静态变量确保单例模式
   - 尝试从注册表加载真实的 CUDA 实现（如果 libATen_cuda.so 已加载）
   - 否则返回默认的 CUDAHooksInterface（所有操作都报错）

3. **注册机制**
   - CUDAHooksRegistry 允许真实的 CUDA 实现（在 libATen_cuda.so 中）注册自己
   - `REGISTER_CUDA_HOOKS` 宏用于注册

### 内存泄漏注解

代码故意泄漏 CUDA hooks 对象，因为：
- 全局对象（如 JIT 内核缓存）可能在程序销毁时需要访问 CUDA hooks
- 如果 CUDA 库先卸载，访问已销毁的对象会导致崩溃
- 泄漏一个虚指针的开销微不足道

### 功能列表

- 初始化 CUDA 状态
- 获取默认/新建随机数生成器
- 查询设备信息（是否有 CUDA、CUDART、cuDNN、cuBLASLt 等）
- 获取设备指针对应的设备
- 检查内存是否固定（pinned）
- 获取固定内存分配器和 CUDA 设备分配器
- 查询 cuDNN/CUDART 版本和配置
- 管理 cuFFT 计划缓存
- 设备同步

---

**核心概念：**
- 动态分发 CUDA 功能，避免编译时依赖
- 优雅降级 - 无 CUDA 库时友好的错误提示
- 单例模式管理全局 CUDA hooks 对象
- 注册表模式支持运行时插件注入
