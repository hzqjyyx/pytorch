这个文件非常简洁，只有3行代码：

```cpp
#pragma once
#include <c10/core/TensorOptions.h>
```

**主要功能：**

- **头文件保护** - `#pragma once` 防止重复包含
- **转发包含** - 将 `c10/core/TensorOptions.h` 中的定义重新导出到 `ATen` 命名空间
- **API 兼容性** - 允许代码通过 `aten/src/ATen/TensorOptions.h` 路径访问张量选项配置

实际的 `TensorOptions` 类定义在 `c10/core/TensorOptions.h` 中，这个文件只是一个包装层。
