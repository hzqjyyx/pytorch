## FuncTorchTLS 文件分析

这两个文件实现了一个**线程局部存储（Thread-Local Storage, TLS）机制**，用于在 PyTorch 主库和 functorch（一个独立维护的库）之间传递状态信息。

### 核心设计

**问题背景**：functorch 是一个独立的out-of-tree库，但需要在 PyTorch 中存储和传递某些TLS状态。为了避免直接依赖，采用了基类指针的间接方案。

**架构**：
- `FuncTorchTLSBase`：PyTorch 中定义的抽象基类（仅包含虚函数接口）
- `FuncTorchTLSImpl`：functorch 库中实现的具体子类（包含实际的元数据，如 DynamicLayerStack）
- PyTorch 存储 `FuncTorchTLSBase*` 指针，但实际指向 functorch 创建的 `FuncTorchTLSImpl` 对象

### 文件职责

**FuncTorchTLS.h**（头文件）：
- 定义 `FuncTorchTLSBase` 虚基类
- 声明三个全局API函数的接口

**FuncTorchTLS.cpp**（实现文件）：
- 维护 `thread_local std::unique_ptr<FuncTorchTLSBase> kFuncTorchTLS`（线程本地全局TLS）
- 实现三个API函数

### 关键函数

1. **`getCopyOfFuncTorchTLS()`**
   - 返回当前TLS的深拷贝
   - 若TLS为空则返回nullptr

2. **`setFuncTorchTLS()`**
   - 设置新的TLS状态
   - 总是进行深拷贝（不直接存储传入的指针）
   - 支持传入nullptr来清除TLS

3. **`functorchTLSAccessor()`**
   - 返回对线程本地TLS的可变引用
   - 允许直接修改当前线程的TLS

### 核心特性

- **线程隔离**：每个线程拥有独立的TLS副本
- **深拷贝语义**：避免跨线程的指针共享问题
- **虚函数支持**：包含 checkSupports* 系列方法用于运行时检查functorch特性支持情况

### 主要用途

- **动态层栈管理**：functorch 的 vmap/grad 等函数式编程特性需要维护计算图信息
- **自动求导约束检查**：在functorch上下文中验证某些自动求导操作是否被支持
- **线程安全的状态传播**：确保多线程环境下functorch状态不会混淆

---

### 总结

- 提供 PyTorch ↔ functorch 之间的**线程局部状态传递接口**
- 通过**虚基类指针**实现库间解耦
- 使用**深拷贝**保证线程安全性
- 支持**运行时特性检查**（自动求导约束验证）
