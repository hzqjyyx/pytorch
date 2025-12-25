# Type_demangle.cpp 功能分析

这个文件实现了C++名称的反混淆（demangling）功能。

## 核心功能

`demangle()` 函数接收一个被编译器混淆的C++函数名，使用 `abi::__cxa_demangle()` 将其转换为可读的格式。

**示例**：`_Z1gv` → `g()`

## 实现细节

- 使用 `cxxabi.h` 中的标准库函数进行反混淆
- 返回值由 `std::unique_ptr` 配合 `free` 作为删除器管理，因为 `__cxa_demangle` 返回的是 malloc 分配的内存
- 反混淆失败时（status != 0）返回原始名称作为降级处理

## 适用场景

- 函数名不遵循标准C++ Itanium ABI混淆方案时失败（如 `main`、`clone`）
- 此时使用原始名称作为合理的默认值

---

## 关键特点

- **条件编译**：仅在 `HAS_DEMANGLE` 为真时编译
- **内存安全**：采用RAII模式处理malloc内存
- **容错设计**：反混淆失败不会导致程序崩溃
