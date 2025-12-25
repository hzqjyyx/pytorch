# c10/util/Backtrace 功能解析

## 核心功能

提供跨平台的堆栈回溯（backtrace）捕获和符号化功能，用于调试和错误报告。

## 主要 API

### `get_backtrace()`
```cpp
std::string get_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64,
    bool skip_python_frames = true);
```
- **已废弃**，立即捕获并符号化堆栈
- 返回格式化的堆栈跟踪字符串

### `get_lazy_backtrace()`
```cpp
Backtrace get_lazy_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64,
    bool skip_python_frames = true);
```
- **推荐使用**，捕获堆栈但延迟符号化
- 返回 `std::shared_ptr<const LazyValue<std::string>>`
- 只在实际需要时才进行昂贵的符号化操作

## 平台实现

### FBCODE_CAFFE2 (c10/util/Backtrace.cpp:41-60)
使用 Facebook 内部的 `facebook::process::StackTrace`，性能更优。

### Android + SUPPORTS_BACKTRACE (c10/util/Backtrace.cpp:62-119)
- 使用 `_Unwind_Backtrace()` 捕获堆栈帧
- 通过 `dladdr()` 获取符号信息
- 使用 `__cxxabiv1::__cxa_demangle()` 解码 C++ 符号名
- 输出格式: `frame #N <demangled_name>[address]`

### Linux/Unix + SUPPORTS_BACKTRACE (c10/util/Backtrace.cpp:121-288)
- 使用 `backtrace()` 获取返回地址数组 (c10/util/Backtrace.cpp:224)
- 使用 `backtrace_symbols()` 获取符号字符串 (c10/util/Backtrace.cpp:243)
- 解析帧信息：
  - **GLIBCXX**: `<object-file>(<mangled-name>+<offset>) [<address>]` (c10/util/Backtrace.cpp:152-181)
  - **LIBCXX**: `<frame#> <object-file> <address> <mangled-name> + <offset>` (c10/util/Backtrace.cpp:182-191)
- Python 帧过滤: 检测 `python`、`python3`、`libpython` (c10/util/Backtrace.cpp:137-140, 258-263)
- 输出格式: `frame #N: <function_name> + <offset> (<address> in <object_file>)`

### Windows (c10/util/Backtrace.cpp:290-436)
- 使用 `CaptureStackBackTrace()` 捕获堆栈 (c10/util/Backtrace.cpp:364)
- 使用 `SymFromAddr()` 获取符号信息 (c10/util/Backtrace.cpp:393)
- 使用 `SymGetLineFromAddr64()` 获取文件名和行号 (c10/util/Backtrace.cpp:401)
- 通过 `GetModuleHandleExW()` 和 `GetModuleFileNameW()` 获取模块名 (c10/util/Backtrace.cpp:298-305)
- 单例 `SymbolHelper` 管理符号初始化 (c10/util/Backtrace.cpp:318-343)
- 输出格式: `<address> <symbol_address> <module>!<symbol_name> [<file> @ <line>]`

### 不支持的平台 (c10/util/Backtrace.cpp:438-450)
返回 `"(no backtrace available)"`

## 关键特性

1. **延迟符号化**: `LazyBacktrace` 类继承 `OptimisticLazyValue<std::string>` (c10/util/Backtrace.cpp:469-479)，只在访问时调用 `symbolize()`
2. **帧跳过**: 自动跳过 backtrace 函数自身帧 (c10/util/Backtrace.cpp:218, 361)
3. **Python 帧过滤**: 可选择性隐藏 Python 解释器内部帧
4. **符号解码**: 将 C++ mangled 名称转换为可读形式

## 不涉及的内容
- ROCm 相关功能（本文件无相关代码）
- Backward 库集成（使用的是系统原生 backtrace API）
