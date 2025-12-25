# SmallVector 核心功能

这是一个优化的动态数组容器，源自 LLVM，针对"通常很小"的数组场景进行了优化。

## 主要设计思想

**小对象优化（Small Object Optimization）**：在对象内部预留一块栈上内存用于存储少量元素，避免小数组频繁的堆分配。只有当元素数量超过内联容量时才会动态分配内存。

## 核心组件

### 1. SmallVectorBase (c10/util/SmallVector.h:52-101)
基础类，管理所有 SmallVector 共有的状态：
- `BeginX`：指向数据起始位置（可能指向内联存储或堆内存）
- `Size`：当前元素数量
- `Capacity`：当前容量

使用模板参数 `Size_T` 控制 size/capacity 字段的类型（uint32_t 或 uint64_t），针对不同元素类型优化内存占用。

### 2. SmallVectorSizeType (c10/util/SmallVector.h:104-105)
自动选择大小类型的策略：
- 对于小于 4 字节且在 64 位平台的类型，使用 uint64_t（避免容量限制，如 `SmallVector<char>` 需要支持 >4GB）
- 其他情况使用 uint32_t（节省空间）

### 3. SmallVectorTemplateBase (c10/util/SmallVector.h:376-627)
根据元素是否 trivially copyable 特化为两个版本：

**非 trivial 版本** (c10/util/SmallVector.h:376-486)：
- 使用 placement new 构造元素
- 显式调用析构函数
- 使用 `std::uninitialized_copy/move` 移动元素
- 参数传递使用 `const T&`

**trivial 版本** (c10/util/SmallVector.h:526-627)：
- 使用 `memcpy` 进行批量复制（性能优化）
- 跳过析构
- 根据元素大小决定参数传递方式：`sizeof(T) <= 2 * sizeof(void*)` 时按值传递（避免引用失效开销）

### 4. SmallVectorImpl (c10/util/SmallVector.h:632-1027)
实现所有容器操作，与内联大小 N 无关，减少代码膨胀：

**关键方法**：
- `push_back/pop_back`：基本操作
- `insert/erase`：插入删除，处理迭代器失效
- `append/assign`：批量操作
- `reserve/resize`：容量管理
- `swap`：优化的交换（非小对象直接交换指针）

**引用失效处理**：
- `reserveForParamAndGetAddress` (c10/util/SmallVector.h:429-437)：检测参数是否指向内部存储，扩容前保存索引，扩容后重新计算地址
- `isSafeToReferenceAfterResize` (c10/util/SmallVector.h:178-189)：验证引用在调整大小后是否有效

### 5. SmallVectorStorage (c10/util/SmallVector.h:1184-1193)
内联存储空间：
- 使用 `alignas(T)` 确保对齐
- N=0 时特化为空类（但仍保持对齐，确保指针运算合法）

### 6. SmallVector (c10/util/SmallVector.h:1274-1405)
最终用户接口，继承 `SmallVectorImpl<T>` 和 `SmallVectorStorage<T, N>`：
- 默认 N 值计算：使整个对象大小约为 64 字节（c10/util/SmallVector.h:1207-1253）
- 提供各种构造函数（范围、容器、初始化列表等）

## 实现细节

### 增长策略 (c10/util/SmallVector.cpp:87-106)
```cpp
NewCapacity = 2 * OldCapacity + 1
```
- 指数增长，摊销 O(1) 插入
- +1 确保从 0 开始也能增长
- 限制在 `Size_T` 的最大值

### 内存管理 (c10/util/SmallVector.cpp:110-151)

**mallocForGrow**：分配新内存，失败抛出 `std::bad_alloc`

**grow_pod** (POD 类型优化)：
- `BeginX == FirstEl`：仍在使用内联存储 → `malloc` 新内存并 `memcpy`
- 否则：已在堆上 → `realloc` 调整大小（可能原地扩展，避免拷贝）

**非 POD 类型增长** (c10/util/SmallVector.h:490-519)：
1. `mallocForGrow`：分配新内存
2. `moveElementsForGrow`：移动构造元素到新位置，析构旧元素
3. `takeAllocationForGrow`：释放旧内存（如果不是内联存储），更新指针

### 对齐处理 (c10/util/SmallVector.h:108-132)
`SmallVectorAlignmentAndSize` 确保：
- Base 部分对齐到 `SmallVectorBase` 要求
- FirstEl 对齐到元素类型 T 要求
- `getFirstEl()` 通过 `offsetof` 计算内联存储的起始地址

### 模板实例化控制 (c10/util/SmallVector.cpp:153-170)
显式实例化：
- `SmallVectorBase<uint32_t>`：32 位平台
- `SmallVectorBase<uint64_t>`：64 位平台且需要大容量时

避免在每个编译单元重复实例化基础方法。

## 与 LLVM 原版的差异

1. 使用 `std::bad_alloc` 而非 `llvm::safe_malloc`
2. 删除了 `LLVM_ENABLE_EXCEPTIONS` 相关代码
3. 使用 C++17 的 `std::is_trivially_*_constructible` 而非 LLVM 的 trait
4. 添加了 `SmallVector::at` 方法（越界断言）
5. 添加了 `operator<<` 支持流输出
6. 用 `C10_API` 导出符号

## 静态断言验证 (c10/util/SmallVector.cpp:41-62)

- 确保 `SmallVector<void*, 0>` 没有浪费空间
- 验证高对齐类型（16B/32B）的对齐和填充正确性
- 小元素类型（如 char）使用合适的 size 类型

---

**ROCm/Backward 相关**：
- 无 ROCm 特定代码
- 无反向传播相关逻辑（这是纯容器实现）
