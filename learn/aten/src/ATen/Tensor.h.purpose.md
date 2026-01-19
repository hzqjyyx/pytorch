这个文件非常简洁，只有4行代码：

```cpp
#pragma once

#include <ATen/core/Tensor.h>
```

**主要功能：**

- **头文件保护** - `#pragma once` 防止重复包含
- **转发包含** - 将 `<ATen/core/Tensor.h>` 的内容暴露给使用者
- **API 简化** - 允许用户通过 `#include <ATen/Tensor.h>` 而不是 `#include <ATen/core/Tensor.h>` 来访问 Tensor 类

这是一个典型的**包装头文件**（wrapper header），用于提供更简洁的包含路径。实际的 Tensor 类定义在 `ATen/core/Tensor.h` 中。
