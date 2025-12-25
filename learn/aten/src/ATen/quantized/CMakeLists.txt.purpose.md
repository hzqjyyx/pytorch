- **文件扫描**：使用 `file(GLOB_RECURSE ...)` 递归扫描目录下的所有文件
  - `ATen_QUANTIZED_HEADERS`：收集所有 `.h` 头文件
  - `ATen_QUANTIZED_SRCS`：收集所有 `.cpp` 源文件
  - `ATen_QUANTIZED_TEST_SRCS`：收集所有 `*_test.cpp` 测试文件

- **排除测试文件**：使用 `EXCLUDE()` 将测试源文件从主源文件列表中移除，避免重复编译

- **向父作用域传递**：使用 `set(...PARENT_SCOPE)` 将收集到的文件列表导出到父级 CMakeLists.txt，供上层构建系统使用
