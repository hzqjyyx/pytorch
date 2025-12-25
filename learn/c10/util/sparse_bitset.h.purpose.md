# SparseBitVector 核心功能

这是一个从 LLVM 移植的稀疏位向量实现，通过只存储包含非零位的元素来节省内存。

## 数据结构设计

**SparseBitVectorElement<ElementSize>** - 底层存储单元：
- 存储固定大小（默认 128 位）的位块
- 使用 `BitWord` 数组（unsigned long）存储实际的位
- 包含 `ElementIndex` 标识该元素在位向量中的位置
- 每个元素负责 ElementSize 个连续的位

**SparseBitVector<ElementSize>** - 主容器：
- 使用 `std::list` 存储稀疏的 Element，只保存有非零位的块
- 维护 `CurrElementIter` 缓存迭代器，优化连续访问性能
- 元素按 ElementIndex 升序排列

## 核心操作

**位操作** (`c10/util/sparse_bitset.h:91-110`):
- `set(Idx)` - 设置位，自动创建新 Element
- `test(Idx)` - 测试位
- `reset(Idx)` - 清除位，空 Element 会被删除

**集合运算**:
- `operator|=` - 并集 (`c10/util/sparse_bitset.h:544`)
- `operator&=` - 交集 (`c10/util/sparse_bitset.h:584`)
- `operator-=` / `intersectWithComplement` - 差集 (`c10/util/sparse_bitset.h:633`)
- `intersects` - 判断是否有交集 (`c10/util/sparse_bitset.h:741`)

**查找操作**:
- `find_first()` / `find_last()` - 查找首/尾设置位 (`c10/util/sparse_bitset.h:777-790`)
- `find_next(Curr)` - 从指定位置查找下一个设置位 (`c10/util/sparse_bitset.h:140`)
- `count()` - 统计设置位数量（使用 popcount）

**迭代器** (`c10/util/sparse_bitset.h:306-426`):
- `SparseBitVectorIterator` 实现正向迭代
- 只遍历设置为 1 的位，跳过 0 位
- 支持标准 C++ 迭代器协议

## 性能优化策略

1. **缓存最近访问位置** - `CurrElementIter` 使连续访问模式达到常数时间
2. **自定义 FindLowerBound** - 从缓存位置开始线性搜索，而非每次二分查找
3. **延迟删除** - 仅当 Element 变为全零时才删除
4. **位级操作** - 使用位运算和 LLVM 的 `countTrailingZeros`/`countPopulation` 等底层函数

## 典型使用场景

在 PyTorch 中用于编译器/图优化中的数据流分析：
- 活跃变量分析
- 到达定义分析
- 依赖关系追踪
- 稀疏索引集合

---

**ROCm 相关**: 无  
**Backward 相关**: 无
