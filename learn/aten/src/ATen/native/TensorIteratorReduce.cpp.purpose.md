## TensorIteratorReduce.cpp 主要功能分析

这个文件实现了 **TensorIterator 中的并行 reduction 操作**，用于在多线程环境下高效执行 reduce 运算。

### 核心函数

**`parallel_reduce()`** (第24-37行)
- TensorIterator 的主入口函数
- 根据数据量和线程情况选择执行策略：
  - 如果数据量小或已在并行区域内：串行执行
  - 否则根据输出大小选择两种并行策略

**`use_two_pass_reduction()`** (第39-41行)
- 判断是否使用两轮 reduce 策略
- 条件：输出张量只有单个元素（标量）

**`two_pass_reduction()`** (第43-80行)
- 两轮 reduce 实现：
  1. 第一轮：各线程独立 reduce 到各自的 buffer 切片
  2. 第二轮：将各线程的 buffer 结果合并到最终输出
- 避免了原子操作，提高并行效率

**`parallel_dim_reduction()`** (第116-138行)
- 沿某一维度的并行 reduce
- 为了缓存局部性，按 128 字节对齐列边界

**`find_split_dim()`** (第84-100行)
- 选择最优的并行分割维度
- 优先选择大于线程数的最外层维度

**`foreach_reduced_elt()`** (第140-193行)
- 遍历所有 reduce 后的元素
- 支持多输出和更复杂的 reduce 模式

### 关键设计

- **避免原子操作**：两轮 reduce 方式消除竞争条件
- **缓存优化**：按内存对齐边界分割列
- **动态策略选择**：根据数据规模和硬件选择最优路径

### 总结

- Reduction 并行化框架，支持标量/向量 reduce
- 两轮算法处理全 reduce，一维分割处理局部 reduce
- 内存对齐优化确保缓存效率
- 处理嵌套并行和线程数动态调整
