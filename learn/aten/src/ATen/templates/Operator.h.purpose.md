这是一个 C++ 模板文件，用于生成 ATen 操作符的声明。主要功能：

- **模板生成文件**：包含 `${generated_comment}` 和 `${declarations}` 占位符，在代码生成过程中被替换
- **前向声明**：通过 `#include <ATen/core/ATen_fwd.h>` 引入必要类型的前向声明，避免循环依赖
- **命名空间组织**：将所有操作符声明放在 `at::_ops` 命名空间下
- **头文件保护**：使用 `#pragma once` 防止重复包含
- **依赖管理**：包含 `<string_view>`、`<tuple>`、`<vector>` 等标准库头文件，为生成的操作符声明提供基础类型支持
