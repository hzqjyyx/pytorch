# DynamicLibrary 功能分析

## 核心功能

这两个文件实现了一个跨平台的动态库加载和符号解析机制。

### DynamicLibrary 类结构

**头文件定义（DynamicLibrary.h）：**
- 继承自 `Error` 的异常类 `DynamicLibraryError`
- `DynamicLibrary` 结构体，禁止复制和移动语义
- 三个公开方法：构造函数、符号查找、析构函数
- 两个私有成员：`leak_handle` 标志和 `handle` 指针

### 实现细节（DynamicLibrary.cpp）

**Unix/Linux 实现（非 Windows）：**
- `dlopen()` 加载动态库，使用 `RTLD_LOCAL | RTLD_NOW` 标志
- 支持主库名和备用库名（如库不存在时尝试备用名）
- `checkDL()` 辅助函数验证操作成功，失败时抛出 `DynamicLibraryError`
- `sym()` 方法通过 `dlsym()` 获取符号地址
- 析构函数通过 `dlclose()` 释放库（除非 `leak_handle` 为真）

**Windows 实现：**
- 使用 `LoadLibraryExW` 和 `LoadLibraryW` 加载库
- 先尝试 `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` 模式（新系统）
- 若不支持则回退到标准 `LoadLibraryW`
- `sym()` 通过 `GetProcAddress()` 获取函数指针
- 析构函数用 `FreeLibrary()` 释放

## 关键特性

- **跨平台抽象**：统一接口隐藏 Unix/Windows 实现差异
- **容错机制**：支持主库名和备用库名，处理库不存在情况
- **符号泄漏控制**：`leak_handle` 标志可选择是否在析构时释放库
- **错误报告**：详细的错误消息（包括系统错误代码）

## 主要用途

- **bullet-point 总结：**
  - 运行时动态加载外部共享库（`.so`、`.dll` 等）
  - 获取库中导出的函数符号地址
  - 支持后备库名，提高兼容性
  - 自动管理库的生命周期（可选手动控制）
  - 为 PyTorch 的跨平台库加载提供基础设施
