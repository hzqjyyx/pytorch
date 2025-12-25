## 文件功能分析

这个文件是一个基准测试文件，用于比较 `intrusive_ptr`（PyTorch 自定义智能指针）和标准库 `std::shared_ptr` 的性能差异。

**核心对比对象：**
- `Foo` 类：继承自 `intrusive_ptr_target`，用于 intrusive_ptr 测试
- `Bar` 类：继承自 `std::enable_shared_from_this<Bar>`，用于 shared_ptr 测试

**四个主要基准测试：**

- **BM_IntrusivePtrCtorDtor** vs **BM_SharedPtrCtorDtor**
  - 测试指针的构造和析构性能
  - 重复复制同一个指针，衡量 copy 操作开销

- **BM_IntrusivePtrArray** vs **BM_SharedPtrArray**
  - 测试批量存储指针的性能
  - 在数组中写入和重置指针，范围从 16 到 4096 个元素
  - 使用 RangeMultiplier(2) 逐倍增加测试规模

- **BM_IntrusivePtrExclusiveOwnership** vs **BM_SharedPtrExclusiveOwnership**
  - 测试独占所有权场景的性能
  - 反复创建和销毁指针对象

**关键特点：**

- 使用 Google Benchmark 框架进行性能测试
- 采用 volatile 变量防止编译器优化掉测试代码
- 通过直接对比量化 intrusive_ptr 相对于 shared_ptr 的性能优势
