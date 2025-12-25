## 文件分析

这两个文件提供了一个跨平台的线程安全错误字符串获取函数。

**error.h** 声明了公共 API：
- `str_error(int errnum)` 函数，接收错误代码，返回对应的错误描述字符串

**error.cpp** 实现了该函数的核心逻辑：
1. 保存当前 `errno` 值（防止函数执行中被修改）
2. 分配 256 字节的缓冲区
3. 根据平台调用不同的系统函数：
   - **Windows**：使用 `strerror_s()` 获取错误信息，然后调整缓冲区大小
   - **Unix/Linux**：使用 `strerror_r()`（可重入版本），处理两种返回类型（int 或 char*）
4. 恢复原始 `errno` 值
5. 返回错误字符串

---

**主要特点：**
- 线程安全（使用可重入函数 `strerror_r` 和 `strerror_s`）
- 跨平台支持（Windows 和 Unix-like 系统）
- 防止 `errno` 被污染
- 使用 `[[maybe_unused]]` 处理可能未使用的返回值
