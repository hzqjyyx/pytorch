这个 Bazel 文件实现了一个自定义规则，模拟 CMake 的 `configure_file` 功能。

**工作原理：**

规则通过 shell 命令处理模板文件：

1. **读取模板** - 使用 `cat` 命令读入源文件
2. **替换定义** - 对 `definitions` 列表中的每个标识符，执行 sed 替换，将 `#cmakedefine FOO` 转换为 `#define FOO`
3. **处理未定义项** - 使用正则表达式将剩余的 `#cmakedefine` 替换为注释形式 `/* #undef FOO */`
4. **输出** - 重定向结果到输出文件

**关键特性：**

- **Bazel 兼容性** - 使用 `attr.string_list` 而非 `dict`，以支持 `select()` 和 `config_setting`（字典不支持 append）
- **头文件处理** - 返回 `CcInfo` provider，使编译器能找到生成的头文件
- **Include 路径** - 将输出目录和 bin 目录都添加到包含路径

**关键要点：**

- 模拟 CMake 预处理器的功能
- 处理条件编译的 `#cmakedefine` 指令
- 将生成的头文件集成到 C++ 编译流程中
- 支持动态配置，适配不同编译条件
