## 主要功能

这两个文件实现了跨平台的临时文件和临时目录创建功能。

### 核心概念

**两个主要类：**
- `TempFile` - 代表临时文件，包含文件描述符和路径名
- `TempDir` - 代表临时目录，包含目录路径名

两个类都禁用拷贝构造和拷贝赋值，仅支持移动语义。析构时自动清理资源。

### 关键实现差异（Unix vs Windows）

**Unix/Linux:**
- 使用 `mkstemp()` 创建临时文件，返回有效的文件描述符
- 使用 `mkdtemp()` 创建临时目录
- 支持环境变量 `TMPDIR`, `TMP`, `TEMP`, `TEMPDIR` 指定临时目录，默认 `/tmp`
- 析构时调用 `unlink()` 和 `close()` 清理

**Windows:**
- 使用 `tmpnam_s()` 生成临时文件名，但不实际创建文件
- 提供 `TempFile::open()` 方法延迟打开文件（使用 `_sopen_s()`）
- 创建临时目录使用 `CreateDirectoryA()`，重试最多 10 次
- 析构时调用 `_close()` 和 `RemoveDirectoryA()`

### API 函数

- `try_make_tempfile()` - 尝试创建临时文件，失败返回 `nullopt`
- `make_tempfile()` - 创建临时文件，失败抛异常
- `try_make_tempdir()` - 尝试创建临时目录，失败返回 `nullopt`
- `make_tempdir()` - 创建临时目录，失败抛异常

### 核心特性

- **自动清理** - RAII 模式，对象析构自动删除文件/目录
- **错误处理** - 提供两种 API：安全版本（返回 `optional`）和异常版本
- **平台适配** - 统一接口下的差异化实现
- **环境变量支持** - Unix 支持自定义临时目录位置

---

- 实现临时文件和临时目录的创建与管理
- 跨平台支持（Unix/Linux 和 Windows）
- RAII 模式自动清理临时资源
- 提供安全版本（`try_*`）和异常版本（`make_*`）两套 API
- Unix 使用 `mkstemp`/`mkdtemp`，Windows 使用 `tmpnam_s`/`CreateDirectoryA`
