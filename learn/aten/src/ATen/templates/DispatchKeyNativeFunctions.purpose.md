这两个文件是 PyTorch ATen 库中用于生成调度键（Dispatch Key）相关的本地函数代码的模板文件。

## 文件结构分析

**DispatchKeyNativeFunctions.h** 是头文件模板，定义：
- 生成一个结构体 `${class_name}`，包含该调度键的所有本地函数声明
- 使用命名空间包装 (`${namespace_prologue}` 和 `${namespace_epilogue}`)
- 禁用 clang-format，因为外部后端可能有不同的格式配置

**DispatchKeyNativeFunctions.cpp** 是实现文件模板，包含：
- 必要的头文件包含
- 一个匿名命名空间，存放辅助函数 (`${helper_fns}`)
- 本地函数的实际定义 (`${native_function_definitions}`)
- 命名空间包装

## 功能概述

- **代码生成模板**：这些文件被代码生成工具使用，根据模板变量（如 `${class_name}`、`${dispatch_declarations}` 等）动态生成针对特定调度键的本地函数接口和实现
- **调度系统支持**：每个调度键（如 CPU、CUDA、Metal 等）都对应一组本地函数，这些模板用于组织这些函数
- **后端独立性**：支持外部后端在自己的代码树中生成相应文件

## 核心要素

- **${class_name}** - 调度键对应的类名
- **${dispatch_declarations}** - 函数声明列表
- **${native_function_definitions}** - 函数实现体
- **${helper_fns}** - 辅助函数（放在匿名命名空间）
- **${includes}** 和 **${native_functions_include}** - 依赖的头文件

---

• 模板文件用于动态生成调度键相关的本地函数代码
• 头文件定义函数接口，实现文件包含具体实现
• 支持多后端架构，每个调度键可独立生成对应代码
• 采用命名空间隔离和匿名命名空间存放辅助函数
• 禁用自动格式化以兼容外部后端的代码风格
