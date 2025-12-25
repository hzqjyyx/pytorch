## ThreadLocal.h 文件分析

这个文件提供了一个跨平台的线程局部存储（Thread Local Storage, TLS）解决方案，针对 Android 平台上 libgnustl 对 `thread_local` 限定符的处理问题。

### 核心问题

Android NDK r17 及更早版本中，`thread_local` 关键字在处理复合类型时存在 bug。该文件通过条件编译提供两种实现方案。

### 两种实现方式

**1. Android 旧版本 (C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)**
- 基于 `pthread_key_*` API 的自定义实现
- 使用 pthread 键管理每个线程的数据
- 支持智能指针语义（`operator*`, `operator->`）
- 在构造时创建 pthread key，析构时清理

**2. 其他平台 (标准 C++)**
- 使用标准 `thread_local` 关键字
- 通过 accessor 函数指针模式包装
- 性能更优，代码更简洁

### 便利宏

- `C10_DEFINE_TLS_static(Type, Name)` - 定义静态 TLS 变量
- `C10_DECLARE_TLS_class_static(Class, Type, Name)` - 类内 TLS 声明
- `C10_DEFINE_TLS_class_static(Class, Type, Name)` - 类内 TLS 定义

### 关键特性

- **智能指针语义** - 支持 `*` 和 `->` 操作符
- **自动内存管理** - RAII 模式，自动清理
- **类型安全** - 模板化设计
- **平台适配** - 自动选择最优实现方案

### 使用示例

```cpp
C10_DEFINE_TLS_static(std::string, str_tls_);
*str_tls_ = "abc";
assert(str_tls_->length() == 3);
```

---

- 跨平台线程局部存储封装，解决 Android 兼容性问题
- 提供两种实现：Android 兼容版（pthread 基）和标准版（thread_local 基）
- 支持复合类型，具有智能指针语义
- 通过宏提供便捷的定义和声明接口
- RAII 模式自动管理生命周期
