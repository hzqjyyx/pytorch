这个文件定义了 C10 macros 库的 Bazel 构建目标：

**主要功能：**

- **cc_library target ("macros")**：编译 macros 库，包含 `Macros.h` 和 `Export.h` 两个公共头文件
  - 链接为静态库 (`linkstatic = True`)
  - 定义宏 `C10_BUILD_MAIN_LIB`
  - 可见性为公开

- **cmake_configure_file target ("cmake_macros_h")**：从模板 `cmake_macros.h.in` 生成 `cmake_macros.h`
  - 条件性定义编译宏：
    - 总是定义：`C10_BUILD_SHARED_LIBS`、`C10_USE_MSVC_STATIC_RUNTIME`
    - 根据配置定义：`C10_USE_GFLAGS`（若启用）、`C10_USE_GLOG`（若启用）

- **filegroup target ("headers")**：收集目录下所有 `.h` 头文件，供上层包引用
