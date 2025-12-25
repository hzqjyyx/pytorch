# SymInt 核心功能分析

## 设计目标

SymInt 是一个用于表示**符号整数**的类，核心目的是支持 PyTorch 的动态形状推导（dynamic shape computation）。它可以表示两种值：
- 具体的 int64_t 值
- 符号化的整数表达式（通过 SymNode 类型擦除）

这使得算子内核可以在不固化具体尺寸的情况下进行形状计算和追踪。

## 内存布局优化

SymInt 采用**指针打包（pointer packing）**技术，仅占用一个字（word）空间，实现了零开销的抽象：

### 编码方案（基于符号位模式）
```
0b0...  → 正整数（直接存储）
0b11... → 小负整数（直接存储）
0b10... → 堆分配的符号节点指针
```

- `MAX_UNREPRESENTABLE_INT` 到 `min_representable_int()` 之间的大负数需要堆分配
- `is_heap_allocated()` 通过 `check_range()` 快速判断，仅需一次比较操作
- 移动端（C10_MOBILE）编译时优化：始终返回 false，消除死代码

### 关键位掩码常量
- `MASK`: 0b111000...（高 3 位）
- `IS_SYM`: 0b101000...（指针标记）
- 指针恢复时需要符号扩展（SymInt.h:92-94）

## 核心操作实现

### 1. 构造与生命周期管理

**promote_to_negative()** (SymInt.cpp:15-21)
- 处理大负数：将其包装为 ConstantSymNodeImpl 节点
- 通过移动语义避免引用计数增加
- 保持不变式：SymInt 内部状态始终有效

**拷贝构造/赋值** (SymInt.h:58-87)
- 堆分配情况：通过 toSymNode() 增加引用计数
- 非堆分配：直接拷贝 data_
- 移动操作：窃取资源并将源置零

**析构** (SymInt.h:123-125)
- 调用 release_() 释放堆节点
- 通过 SymNode::reclaim() 管理引用计数

### 2. 二元运算符

**DEFINE_BINARY 宏** (SymInt.cpp:45-62)
实现了 4 种组合的分派逻辑：
1. 两个具体值 → 直接计算（std::plus, std::minus 等）
2. 具体值 + 符号值 → 包装具体值为节点后调用符号运算
3. 符号值 + 具体值 → 调用节点的 METHOD（如 add, mul）
4. 两个符号值 → 直接调用节点方法

支持的运算：
- 算术：`+, -, *, /, %`
- 比较：`sym_eq, sym_ne, sym_lt, sym_le, sym_gt, sym_ge`（返回 SymBool）
- 极值：`min, max`

### 3. 类型转换

**operator SymFloat()** (SymInt.cpp:78-84)
- 具体值：直接转换为 double
- 符号值：调用 SymNodeImpl::sym_float()

### 4. 值提取与守卫

**maybe_as_int()** (SymInt.h:233-242)
分三层尝试：
1. 非堆分配 → 直接返回 data_
2. ConstantSymNodeImpl → 返回常量值
3. 符号节点 → 尝试 maybe_as_int()（可能失败）

**guard_int()** (SymInt.cpp:118-124)
- 插入守卫条件，将符号值固化为具体值
- 用于必须使用具体值的场景（但会导致过度特化）
- 文件名/行号用于诊断

**expect_size()** (SymInt.cpp:126-132)
- 检查值是否为非负数（size-like）
- 对 unbacked SymInt 特殊处理（假设 >= 2）

### 5. 一元运算

**operator-** (SymInt.cpp:134-154)
- 具体值：使用 `__builtin_sub_overflow` 检测溢出
- 溢出时返回原值（避免未定义行为）
- 符号值：调用 neg() 方法

## 与标量类型的互操作

**DEFINE_SYMINT_OP 宏** (SymInt.cpp:196-264)
为 int32_t, int64_t, uint32_t, uint64_t, float, double, size_t 生成：
- 混合算术运算符（SymInt OP scalar 和 scalar OP SymInt）
- 比较运算符（返回 bool）
- 通过 Convert 模板避免不必要的类型转换

## 辅助功能

### is_same() (SymInt.cpp:86-100)
语义同一性检查：
- 堆分配状态不同 → false
- 都非堆分配 → 值相等性检查
- 都堆分配 → 指针相等性检查（共享节点）

### clone() (SymInt.cpp:110-116)
深拷贝：
- 具体值直接返回新 SymInt
- 符号节点调用 SymNodeImpl::clone()

### wrap_node() (SymInt.cpp:102-108)
将 SymInt 规范化为 SymNode：
- 具体值通过 base->wrap_int() 包装
- 符号值返回现有节点

### toSymNodeImplUnowned() (SymInt.h:89-98)
指针解码：
1. 提取低 62 位（掩码 ~MASK）
2. 符号扩展至 64 位
3. 转换为 SymNodeImpl*

## 工具函数

**multiply_integers()** (SymInt.h:294-318)
- 容器版本和迭代器版本
- 使用 std::accumulate 累乘
- 初始值为 SymInt(1)

## 不变式与约束

1. **类型不变式**: `is_heap_allocated() → toSymNodeImplUnowned()->is_int() == true`
2. **范围约束**: `[-2^63, -2^62-1]` 不可直接表示为具体值
3. **生命周期**: 移动后源对象 data_ 置零，防止重复释放
4. **平台优化**: 移动端永不堆分配，编译器可消除符号化路径

---

**ROCm/Backward 相关**: 无相关内容  
**特殊处理**: macOS 需单独声明 size_t 运算符（与 uint64_t 类型不同）
