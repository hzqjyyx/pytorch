这个文件提供了跨平台的宏定义，用于标记代码声明为已弃用（deprecated）。

**主要功能：**

- **C10_DEPRECATED** - 基础弃用宏，用于标记函数、结构体等声明为已弃用
  - C++14+ 使用 `[[deprecated]]` 属性
  - GCC 使用 `__attribute__((deprecated))`
  - MSVC 使用 `__declspec(deprecated)`

- **C10_DEPRECATED_MESSAGE(message)** - 带自定义消息的弃用宏，编译器会显示指定的警告信息

- **C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy)** - 用于标记 `using` 类型别名声明为已弃用
  - 由于 `using` 声明的特殊性，无法直接应用 `[[deprecated]]` 属性
  - 根据编译器支持情况选择合适的实现方式
  - MSVC + CUDA 环境下无法实现，直接定义为普通别名

- **编译器兼容性处理** - 自动检测编译器类型并选择相应的实现方案
  - MSVC：优先使用 `[[deprecated]]`（C++14+），否则用 `__declspec(deprecated)`
  - GCC/Clang：使用 `__attribute__((deprecated))`
  - 其他编译器：发出警告并提供空实现
