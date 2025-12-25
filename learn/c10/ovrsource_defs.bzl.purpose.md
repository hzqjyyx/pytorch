# c10/ovrsource_defs.bzl 文件分析

这是一个 Bazel 构建定义文件，用于在 Meta 的 OVRSource 构建系统中编译 PyTorch 的 c10 库。

## 核心功能

**平台支持定义：**
- CPU 平台：Android、iOS、Linux、macOS、Windows、ARM64
- CUDA 平台：Linux CUDA、Windows CUDA

**主要函数：**

1. **`define_c10_ovrsource(name, is_mobile)`**
   - 编译 c10 CPU 库的通用函数
   - 根据 `is_mobile` 标志设置编译器标志（移动版本添加 `-DC10_MOBILE=1`）
   - 包含源文件：`core/*.cpp`、`util/*.cpp`、`mobile/*.cpp`
   - 配置编译警告抑制（MSVC、Clang 的不同设置）
   - 导出公共头文件和依赖（gflags、cpuinfo、fmt、glog）

2. **`define_ovrsource_targets()`**
   - 生成 CMake 宏头文件（区分移动/非移动版本）
   - 创建两个 c10 库变体：
     - `c10_mobile_ovrsource`：移动设备版本
     - `c10_full_ovrsource`：完整版本
   - 创建 `c10_cuda_ovrsource`：CUDA 加速库
   - 配置条件依赖（Android/iOS 使用移动版本，其他使用完整版本）

**关键特性：**
- 条件编译：使用 `select()` 根据目标平台选择不同配置
- 头文件管理：分离公共头文件和内部实现头文件
- 灵活的编译器标志：针对不同编译器（MSVC、Clang）自适应
- 跨平台支持：统一的构建定义适配多个操作系统和硬件架构

---

## 总结

- 为 OVRSource 系统定义 c10 库的编译规则
- 支持 CPU（移动/完整）和 CUDA 两条构建路径
- 通过 CMake 宏头文件管理平台特定的功能开关
- 处理跨平台编译器兼容性和警告抑制
