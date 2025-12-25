# CUDAContextLight.h 主要功能

这是一个轻量级的 CUDA 上下文头文件，用于减少编译时的传递包含（transitive includes）。

## 核心内容

**包含依赖：**
- 基础 CUDA 运行时库（cuda_runtime_api.h）
- 稀疏矩阵库（cusparse.h）
- BLAS 库（cublas_v2.h, cublasLt.h）
- 线性代数求解库（cusolverDn.h，条件编译）
- 其他 CUDA 相关库

**声明的接口函数：**
- 设备管理：getNumGPUs()、is_available()、getCurrentDeviceProperties()、getDeviceProperties()、canDeviceAccessPeer()
- 内存分配：getCUDADeviceAllocator()
- 句柄管理：getCurrentCUDASparseHandle()、getCurrentCUDABlasHandle()、getCurrentCUDABlasLtHandle()、getCurrentCUDASolverDnHandle()
- 工作空间清理：clearCublasWorkspaces()

## 主要特点

- **设计目的**：提供 CUDA-only 文件使用的统一接口，与 CUDAHooks（用于 CPU 和 CUDA 混合构建）区分
- **状态管理**：只定义接口，不定义类，实际状态由各模块自己管理
- **单一上下文**：全局只有一个 CUDA 上下文/状态

**关键特性：**
- • 轻量级头文件，减少编译依赖
- • 统一的 CUDA 功能接口
- • 支持条件编译（CUDART_VERSION、USE_CUDSS）
- • CUDA 设备属性和能力查询
- • CUDA 库句柄的中央管理
- • 内存分配器接口暴露
