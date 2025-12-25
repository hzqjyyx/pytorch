ssize.h 提供了 C++20 标准库中 `std::ssize()` 函数的实现。该函数返回容器的有符号整数大小。

**核心功能：**

- **避免符号比较警告**：通过返回有符号类型（`std::ptrdiff_t`），避免在比较容器大小时触发 `-Werror=sign-compare` 编译器警告

- **两个重载版本**：
  - 适用于有 `size()` 方法的容器（如 vector、string 等）
  - 适用于 C 风格数组

- **类型转换**：返回类型是 `std::ptrdiff_t` 和容器 `size()` 返回类型的有符号版本的公共类型

- **溢出检查**：使用 `TORCH_INTERNAL_ASSERT_DEBUG_ONLY` 在调试模式下检查是否会发生整数溢出，仅在调试构建中产生性能开销

- **参数相关查找（ADL）**：支持容器通过在自身命名空间中定义自由函数来特化此模板

**使用方式：**
```cpp
using c10::ssize;
auto size = ssize(my_container);  // 返回有符号整数
```
