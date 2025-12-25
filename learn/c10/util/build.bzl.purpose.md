## c10/util/build.bzl 主要功能

这个文件定义了 C10 utility 库的 Bazel 构建目标。通过 `define_targets(rules)` 函数，创建了以下核心库：

- **TypeCast**: 类型转换相关功能，依赖 ScalarType 和基础库
- **base**: 主要工具库，包含除 TypeCast 和 typeid 外的所有 .cpp/.h 文件，依赖 fmt、gflags (可选)、glog (可选)，使用 `-ldl` 链接选项 (非 Windows)
- **bit_cast**: 位转换头文件库，内部可见性
- **ssize**: 安全 size 相关头文件库，依赖 base
- **typeid**: 类型ID相关功能，依赖 ScalarType 和基础库
- **base_headers**: 仅包含头文件的公共库 (除 bit_cast.h、ssize.h)
- **headers**: 头文件分组，用于上层构建依赖

所有库都启用了 `C10_BUILD_MAIN_LIB` 定义，`base` 库使用 `alwayslink = True` 确保符号不被链接器移除。
