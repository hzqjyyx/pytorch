## Load.h 功能分析

这个文件提供了从内存地址安全加载值的通用机制。

**核心设计：**

- 定义了 `LoadImpl<T>` 模板结构，通过 `reinterpret_cast` 从 `void*` 指针读取类型为 `T` 的值
- 提供了两个公共 `load()` 函数重载，支持 `void*` 和类型化指针两种调用方式
- 使用 `C10_HOST_DEVICE` 宏使代码同时支持 CPU 和 GPU 执行

**特殊处理：**

- `LoadImpl<bool>` 专门化：为了避免未定义的 bool 值（见 issue gh-54789），先以 `unsigned char` 形式读取，再转换为 bool
- 静态断言验证 `bool` 大小等于 `char`，确保转换安全

**关键特征：**

- 低级内存操作，用于二进制数据反序列化或内存访问
- `constexpr` 关键字允许编译期常量求值
- 头文件只包含必要的依赖（`<c10/macros/Macros.h>` 和 `<cstring>`）

**用途概括：**

- 提供类型安全的内存读取接口
- 处理 bool 类型的特殊情况以防止无效值
- 支持异构计算（CPU/GPU）场景
