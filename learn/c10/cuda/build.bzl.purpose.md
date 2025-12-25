# c10/cuda/build.bzl 文件分析

这个文件定义了 C10 CUDA 库的 Bazel 构建目标。

## 主要功能

**`define_targets()` 函数**：接收构建规则和额外的编译定义，定义三个核心构建目标：

1. **`cuda` 库目标**
   - 编译 `*.cpp` 和 `impl/*.cpp` 中的所有源文件
   - 导出 `*.h` 和 `impl/*.h` 中的头文件（除了 `CUDAMacros.h`）
   - 添加 `USE_CUDA` 编译宏
   - 依赖于 CUDA 工具链（`@cuda`）和其他 c10 子模块
   - 设置 `alwayslink=True` 防止注册的实体被优化器移除

2. **`Macros` 库目标**
   - 单独编译 CUDA 宏定义
   - 使用 CMake 配置文件 `impl/cuda_cmake_macros.h.in` 生成 `impl/cuda_cmake_macros.h`
   - 暴露公共 API

3. **`headers` 文件组**
   - 收集所有头文件供外部包引用
   - 仅限 `//c10` 包内使用

## 关键特性

- **条件编译**：`target_compatible_with = rules.requires_cuda_enabled()` 确保仅在 CUDA 启用时构建
- **静态链接**：`linkstatic=True` 确保库被静态链接到依赖者
- **构建标记**：`C10_BUILD_MAIN_LIB` 标记这是主库构建

## 简化总结

- 核心 CUDA 库编译与链接配置
- CUDA 宏的独立生成和管理
- 公共头文件导出接口
