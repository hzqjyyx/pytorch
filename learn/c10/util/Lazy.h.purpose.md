# c10/util/Lazy.h 文件分析

这个文件实现了三个与延迟计算相关的类：

## OptimisticLazy<T>
- 线程安全的延迟值容器，使用原子操作实现
- 值在堆上分配，优化了从不计算的场景
- `ensure()` 方法：首次访问时调用工厂函数生成值，并发访问时可能多个线程都会调用工厂函数，但只有一个结果被保存
- 支持拷贝构造、移动构造和赋值操作
- 使用 `std::memory_order_acquire/release` 确保内存可见性

## LazyValue<T>
- 纯虚基类，定义延迟值计算的接口
- 提供 `get()` const 方法获取计算后的值

## OptimisticLazyValue<T>
- 继承 `LazyValue<T>`，基于 `OptimisticLazy<T>` 实现
- 子类需实现虚函数 `compute()`
- `get()` 方法通过 lambda 调用 `compute()` 完成延迟计算

## PrecomputedLazyValue<T>
- 继承 `LazyValue<T>`，用于非延迟的预计算值
- 在构造时接收值，`get()` 直接返回存储的值
- 适合接口统一但值已经存在的场景

---

**要点总结：**
- 线程安全的延迟计算容器
- 支持并发首次访问，但只保存一个结果
- 提供接口抽象和具体实现
- 内存优化：值从不计算则不分配堆空间
