这个文件非常简洁，只有两行内容：

```cpp
#include <ATen/core/Formatting.h>
```

**主要功能：**

- **转发头文件** - 这是一个包装器/代理头文件
- **重定向包含** - 将 `ATen/Formatting.h` 的包含请求转发到 `ATen/core/Formatting.h`
- **API 兼容性** - 维持向后兼容的包含路径，允许代码使用 `#include <ATen/Formatting.h>` 而实际引入的是核心实现

实际的格式化功能实现在 `ATen/core/Formatting.h` 中。
