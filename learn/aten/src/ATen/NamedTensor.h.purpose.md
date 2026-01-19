这个文件非常简洁，只有两行内容：

```cpp
#include <ATen/core/NamedTensor.h>
```

**主要功能：**

- 作为头文件包装器，转发包含 `ATen/core/NamedTensor.h`
- 提供对命名张量（Named Tensor）核心功能的访问接口
- 属于 ATen 库的公共 API 层

实际的命名张量实现位于 `ATen/core/NamedTensor.h`。如果需要了解具体功能，应该查看那个文件。
