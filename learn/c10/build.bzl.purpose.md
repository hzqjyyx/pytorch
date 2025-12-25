**c10/build.bzl 主要功能：**

该文件是 PyTorch c10 库的 Bazel 构建配置文件，定义了两个主要的 C++ 库目标：

- **c10 库**：主库，包含 CPU 和 CUDA 相关的核心功能
  - 依赖于多个子模块：CPUAllocator、ScalarType、alignment、内存分配、基础库等
  - 通过 `rules.if_cuda()` 条件依赖 CUDA 相关模块（CUDAContextLight、Macros 等）
  - 对外公开（visibility: public）

- **c10_headers 库**：仅包含头文件的库
  - 提供核心基础头文件、宏定义、工具函数
  - 包括类型转换、位操作（bit_cast）、大小计算（ssize）等实用工具
  - 对外公开

**核心作用：**
- 定义 c10 库的模块化编译结构
- 管理 CPU 和 GPU（CUDA）代码的条件编译
- 为上层 ATen 等模块提供基础支持
