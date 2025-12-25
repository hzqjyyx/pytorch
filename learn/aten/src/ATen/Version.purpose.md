# Version.h 和 Version.cpp 功能分析

## 主要功能

这两个文件提供了获取 PyTorch 构建配置和依赖库版本信息的接口。

### 核心函数

**show_config()** - Version.cpp:135
- 生成详细的构建配置字符串，包含编译器信息（GCC、Clang、MSVC）、C++ 版本、各依赖库版本等
- 输出格式为可读的多行文本，用于调试和日志记录

**get_mkl_version()** - Version.cpp:21
- 获取 Intel MKL 库版本信息
- 使用 MKL 提供的 API（mkl_get_version_string）
- 如果未启用 MKL，返回 "MKL not found"

**get_mkldnn_version()** - Version.cpp:37
- 获取 Intel MKL-DNN（现已更名为 oneDNN）版本
- 通过 dnnl_version() 调用获取主版本、次版本、补丁版本和 Git Hash

**get_openmp_version()** - Version.cpp:55
- 根据 _OPENMP 宏定义判断 OpenMP 版本（2.5、3.0、3.1、4.0、4.5）
- 如果未启用 OpenMP，返回 "OpenMP not found"

**get_cpu_capability()** - Version.cpp:93
- 返回 CPU 能力标识（AVX2、AVX512、DEFAULT 等）
- 可通过环境变量覆盖

**get_cxx_flags()** - Version.cpp:221
- 获取编译时使用的 CXX_FLAGS
- 仅在开源版本可用（Fbcode 版本会报错）

---

## 快速总结

- **用途**: 运行时获取 PyTorch 构建配置信息
- **主要调用方**: 诊断、日志、版本检查功能
- **依赖库追踪**: MKL、MKL-DNN、OpenMP、LAPACK、NNPACK、CUDA 等
- **编译器信息**: 记录 GCC、Clang、MSVC 等编译器版本
- **CPU 能力**: 报告构建时使用的 CPU 指令集支持
- **条件编译**: 大量 #if/#ifdef 指令根据构建选项决定功能可用性
