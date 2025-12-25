## CUDAHooksInterface 文件功能解析

这两个文件实现了 PyTorch 的 **CUDA 动态派发机制**，允许 CPU 代码调用 CUDA 功能，同时保持 CPU 和 CUDA 库的独立编译。

### 核心设计原理

**问题背景**：PyTorch 被分为两个共享库 — CPU 库和 CUDA 库。如果 CUDA 库未加载，CPU 代码不应该直接调用 CUDA 代码（避免链接依赖）。

**解决方案**：
- `CUDAHooksInterface` 是一个抽象接口，定义所有可能被 CPU 代码调用的 CUDA 功能
- 提供默认存根实现（stub），当 CUDA 库未加载时返回错误信息
- 通过注册表机制（Registry），CUDA 库可以在运行时注册真实实现，覆盖默认存根

### 关键特性

**内存泄漏的故意设计**（CUDAHooksInterface.cpp:6-20）：
- 函数返回的 CUDA hooks 对象被故意泄漏
- 原因：程序销毁时某些全局对象（如 JIT 内核缓存）可能需要访问 CUDA hooks，但 CUDA 库已卸载
- 泄漏的代价很小（仅一个虚函数指针）

**静态初始化策略**（CUDAHooksInterface.cpp:32-39）：
- 使用 `static` 变量确保单例初始化
- 若 CUDA 库未加载，CUDA 功能永久禁用（无法后续动态加载）

### 接口提供的功能

- 生成器管理（默认生成器、新生成器）
- 设备查询（设备索引、GPU 数量、设备同步）
- 内存管理（Pinned 内存分配器、CUDA 设备分配器）
- 库版本查询（cuDNN、CUDA Runtime 版本）
- 库能力检查（cuDNN、cuBLAS、cuSOLVER、cuFFT 编译支持情况）
- cuFFT 计划缓存管理
- NVRTC（运行时编译）接口

### 主要特点

- **Stub 实现**：所有虚拟方法均抛出 `TORCH_CHECK` 错误（包含诊断提示信息）
- **Registry 模式**：CUDA 库通过 `CUDAHooksRegistry` 注册真实实现
- **平台差异处理**：对 MSVC 和 Unix-like 系统提供不同的链接错误提示
- **Forward declare**：仅前向声明 `at::cuda::NVRTC`，避免编译依赖

### 总结

| 文件 | 职责 |
|------|------|
| `.h` | 定义接口、存根实现、注册机制、错误提示 |
| `.cpp` | 实现 `getCUDAHooks()` 工厂函数，处理动态加载逻辑 |

**核心机制**：
- 延迟绑定（Lazy binding）：CUDA hooks 在首次调用时初始化
- 优雅降级（Graceful degradation）：CUDA 库缺失时给出明确错误信息
- 零运行时开销：无 CUDA 库时仅返回存根对象
