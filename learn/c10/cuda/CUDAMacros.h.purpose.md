## 文件功能分析

这个头文件定义了 CUDA 库的导出宏，用于跨平台的动态链接库符号可见性控制。

### 主要内容：

**条件编译配置**
- 包含 CMake 生成的 CUDA 宏配置文件（`cuda_cmake_macros.h`）
- 通过 `C10_CUDA_NO_CMAKE_CONFIGURE_FILE` 可跳过此包含

**跨平台导出宏定义**
- Windows 下：使用 `__declspec(dllexport/dllimport)`
- GCC/非 Windows：使用 `__attribute__((__visibility__("default")))`
- 通过 `C10_CUDA_BUILD_SHARED_LIBS` 控制是否构建动态库

**API 导出宏**
- `C10_CUDA_EXPORT`：导出符号定义
- `C10_CUDA_IMPORT`：导出符号导入
- `C10_CUDA_API`：最终的 API 符号（取决于 `C10_CUDA_BUILD_MAIN_LIB`）

**编译时常量**
- `C10_COMPILE_TIME_MAX_GPUS`：识别的最大 GPU 数量
  - Fbcode 环境：16（向后兼容 Caffe2）
  - 其他环境：120（上限 128，因为 DeviceIndex 是 uint8_t）

### 核心作用：
- 统一管理 libc10_cuda 库的符号可见性
- 支持共享库和静态库构建方式
- 提供跨平台兼容的导出机制
