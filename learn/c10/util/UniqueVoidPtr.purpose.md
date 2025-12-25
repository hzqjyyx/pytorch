UniqueVoidPtr 是 PyTorch c10 库中的一个自定义智能指针类，专门用于管理数据指针和上下文指针的分离场景。

**主要设计思想：**

标准的 `std::unique_ptr` 假设数据指针和删除器之间是一对一的关系。但在许多场景下（例如张量数据分配），实际需要管理的数据指针（如 float*）与需要释放的上下文指针（如 DLManagedTensor）是不同的。UniqueVoidPtr 通过以下方式解决这个问题：

1. **分离的所有权模型**：
   - `data_` - 非所有权指针，指向实际数据
   - `ctx_` - 所有权指针（`std::unique_ptr`），指向需要删除的上下文

2. **关键的行为差异**（相对于 std::unique_ptr）：
   - 删除器作用在上下文指针而非数据指针
   - 即使 `data_` 为 nullptr，如果 `ctx_` 非空，删除器仍会被调用（避免泄漏）
   - `release_context()` 而非 `release()`，提供完整的生命周期管理信息

**核心功能：**

- `UniqueVoidPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter)` - 创建指针对，指定自定义删除器
- `get()` / `get_context()` - 获取数据指针或上下文指针
- `release_context()` - 释放所有权并返回上下文指针
- `move_context()` - 获取上下文的右值引用
- `compare_exchange_deleter()` - 原子式地更换删除器
- `cast_context<T>()` - 类型安全的上下文强制转换

**核心特性：**

- void 指针通用性，支持任意类型数据
- 函数指针删除器，无虚函数开销
- 保证删除器调用（非空上下文时）
- 编译期零开销抽象

**主要用途：**

- Tensor 数据内存管理
- 跨边界资源所有权转移（如 DLPack 集成）
- 自定义分配器支持
