这个文件定义了用于 JIT 编译器迭代器的宏，主要功能如下：

**核心宏定义：**

- `JITERATOR_HOST_DEVICE`：条件编译宏，在 MSVC + CUDA 环境下为空（避免 NVRTC 编译错误），其他情况下映射到 `C10_HOST_DEVICE`

- `jiterator_code(...)`：包装宏，用 `__VA_ARGS__` 处理内核代码中的逗号，防止预处理器误解

- `stringify_code(...)`：将代码转换为字符串字面量（仅在 CUDA 编译时定义）

- `jiterator_also_stringify_as(code, str_name)`：双重定义宏
  - CUDA 编译：同时定义函数代码和对应的 `std::string` 字符串版本
  - CPU 编译：仅定义函数代码

**使用场景：**

- 允许在 CPU 和 CUDA 中共享相同的内核代码定义
- 为 CUDA JIT 编译生成代码字符串，无需手动维护两份代码
