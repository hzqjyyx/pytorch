## 文件功能分析

这两个文件提供了 Windows 平台上的 UTF-8 和 UTF-16 字符串转换功能。

**Unicode.h** (`c10/util/Unicode.h:1-14`) 定义了两个公开 API 函数的声明，仅在 Windows 平台编译（`_WIN32` 条件编译）：
- `u8u16`: UTF-8 字符串转换为 UTF-16（宽字符）字符串
- `u16u8`: UTF-16 字符串转换回 UTF-8 字符串

**Unicode.cpp** (`c10/util/Unicode.cpp:1-49`) 提供了这两个函数的 Windows 实现，使用 Windows API：

**u8u16 函数** (`c10/util/Unicode.cpp:5-21`)：
- 调用 `MultiByteToMultiWideChar` 两次：第一次计算所需缓冲区大小，第二次执行实际转换
- 转换失败时通过 `TORCH_CHECK` 抛出异常
- 处理空字符串的边界情况

**u16u8 函数** (`c10/util/Unicode.cpp:22-47`)：
- 调用 `WideCharToMultiByte` 两次：先计算大小，再执行转换
- 同样使用 `TORCH_CHECK` 进行错误检查
- 处理空字符串的边界情况

---

## 核心功能总结

- **目的**: Windows 平台字符编码转换（UTF-8 ↔ UTF-16）
- **使用场景**: 处理文件路径、命令行参数等需要 Windows Unicode API 的场景
- **错误处理**: 转换失败时抛出 Torch 异常
- **平台限制**: 仅 Windows 编译（`#if defined(_WIN32)`）
