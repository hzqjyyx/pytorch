# c10/CMakeLists.txt 主要功能

这个文件定义了 PyTorch 的 c10 库的构建配置。c10 是一个轻量级的基础库，被设计成最小化依赖。

## 核心内容：

- **项目基础设置**：C++17 标准、CMake 3.18+、导出编译命令
- **宏配置**：生成 cmake_macros.h，包含编译时选项（GFLAGS、GLOG、NUMA、ROCM_KERNEL_ASSERT 等）
- **源文件收集**：使用 glob 模式收集 .cpp 和 .h 文件，包括核心模块（core/、util/、mobile/、macros/）
- **库创建与编译选项**：创建 c10 库目标，启用隐藏可见性（-fvisibility=hidden）、deprecated 警告、错误检查（sign-compare、shadow）
- **可选工具集成**：Include-What-You-Use（IWYU）用于头文件清理
- **依赖管理**：
  - gflags、glog（可选）
  - fmt、nlohmann（必需）
  - NUMA、cpuinfo（可选）
  - Backtrace（可选）
  - mimalloc（可选内存分配器）
  - Threads、dl（Linux）
  - log（Android）
- **子目录**：test、benchmark、cuda（可选）、xpu（可选）
- **安装规则**：库文件安装到 lib/，头文件安装到 include/c10/，PDB 文件（MSVC）安装到 lib/
