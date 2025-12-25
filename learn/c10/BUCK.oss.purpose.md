**c10/BUCK.oss 文件分析**

这是一个 Bazel/Buck 构建配置文件，定义了 c10 库的编译和链接规则。

**主要功能：**

- **库定义**：声明一个名为 `c10` 的 C++ 库（cxx_library）
- **源文件收集**：通过 glob 模式递归收集所有 `.cpp` 文件，排除 test、benchmark、cuda 和 xpu 目录
- **依赖管理**：引入三个第三方库（cpuinfo、fmt、glog）
- **编译标志**：启用严格编译检查（-Werror）和特定的宏定义（DISABLE_NAMEDTENSOR、DISABLE_BACKTRACE）
- **头文件导出**：公开所有 `.h` 头文件（除 test、benchmark、cuda、xpu），设置命名空间为 `c10`
- **预处理器标志**：配置多个编译时宏（C10_USE_GLOG、C10_MOBILE 等）、异常处理、日志系统
- **平台特定配置**：为 Windows 平台添加特定的预编译标志和宏定义
- **可见性**：将库标记为 PUBLIC，允许其他模块依赖
