## 文件功能分析

这两个文件实现了对 CUDA 驱动 API 和 NVML（NVIDIA Management Library）的动态加载和封装。

### driver_api.h

定义了：
- **C10_CUDA_DRIVER_CHECK 宏**：用于检查 CUDA 驱动调用的返回值，如果失败则获取错误字符串并抛出异常
- **C10_LIBCUDA_DRIVER_API 宏**：列出需要从 libcuda.so.1 中动态加载的驱动函数（内存管理、流操作等）
- **C10_LIBCUDA_DRIVER_API_12030 宏**：CUDA 12.0.3+ 版本特定的函数（多播相关）
- **C10_NVML_DRIVER_API 宏**：列出需要从 libnvidia-ml.so.1 中动态加载的 NVML 函数
- **DriverAPI 结构体**：包含所有上述函数的指针成员，提供统一的接口访问这些驱动函数

### driver_api.cpp

实现了：
- **create_driver_api() 函数**：
  - 使用 `dlopen()` 动态加载 libcuda.so.1 和 libnvidia-ml.so.1
  - 通过 `dlsym()` 查找各个函数符号，填充 DriverAPI 结构体的成员指针
  - 对必需函数进行断言检查，可选函数则忽略查找失败
- **DriverAPI::get_nvml_handle()**：静态函数，返回 NVML 库的句柄（单例模式）
- **DriverAPI::get()**：返回唯一的 DriverAPI 单例实例

### 核心目的

- 在运行时动态加载 CUDA 驱动库，避免编译时的硬依赖
- 提供函数指针的统一管理和访问接口
- 通过错误检查宏确保驱动调用的安全性

### 关键特性

- **动态加载**：使用 dlopen/dlsym 而非静态链接
- **单例模式**：DriverAPI::get() 确保全局只有一个实例
- **版本适配**：支持不同 CUDA 版本的函数集合
- **可选 NVML**：NVML 库加载失败不会导致程序崩溃
