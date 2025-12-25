这个文件提供了 C++ 类型名称的处理和显示功能：

- **`demangle()` 函数**：将 C++ 编译器生成的损坏的符号名称（mangled name）还原为可读的形式
  - 接收 `const char*` 类型的损坏名称
  - 返回 `std::string` 类型的可读名称

- **`demangle_type<T>()` 模板函数**：获取任意类型 T 的可读名称
  - 利用 RTTI（运行时类型信息）获取类型的 typeinfo
  - 调用 `demangle()` 进行还原
  - 使用 `__GXX_RTTI` 宏进行条件编译
  - 当 RTTI 禁用时返回提示信息 `"(RTTI disabled, cannot show name)"`

- **核心用途**：使开发者能以人类可读的形式打印和调试 C++ 类型信息
