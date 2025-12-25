这个文件是 CMake 配置模板，用于生成 `ATenConfig.cmake`，使其他项目能够找到并链接 ATen 库。

主要功能：

- **定义查找变量**: 设置 `ATEN_FOUND`、`ATEN_INCLUDE_DIR`、`ATEN_LIBRARIES` 等 CMake 变量
- **模板替换**: 使用 `@...@` 占位符，在编译时由 CMake 将实际的路径和库列表填充进去
- **标准 CMake 接口**: 遵循 CMake 的 `Find*.cmake` 约定，允许用户通过 `find_package(ATen)` 导入
- **依赖声明**: 声明 ATen 的依赖库（如 CUDA、cuDNN 等，通过占位符注入）
- **跨项目集成**: 使下游项目能够自动获取正确的编译标志和链接选项
