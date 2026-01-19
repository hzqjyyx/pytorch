根据文件内容，这是一个简单的头文件：

**主要功能：**

- 包含预处理指令 `#pragma once`，防止重复包含
- 引入 `<ATen/core/Generator.h>` 头文件
- 作为 ATen 生成器功能的公共接口层

**文件作用：**

- 提供对核心生成器实现的访问
- 充当 `ATen/core/Generator.h` 的包装/转发头文件
