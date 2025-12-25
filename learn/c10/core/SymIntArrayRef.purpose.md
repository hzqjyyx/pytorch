## SymIntArrayRef 文件分析

**SymIntArrayRef.h** 定义了一个类型别名和一组转换函数，用于在 `SymInt`（符号整数）和普通 `int64_t` 之间进行互操作：

### 核心类型定义
- `SymIntArrayRef` = `ArrayRef<SymInt>` - 符号整数数组的非所有权引用

### 主要转换函数

**SymInt → IntArrayRef 方向：**
- `asIntArrayRefUnchecked()` - 直接重新解释转换，不做检查（最快，假设数据有效）
- `asIntArrayRefSlowOpt()` - 检查是否包含堆分配的大整数，包含则返回 `nullopt`
- `asIntArrayRefSlow()` - 检查所有元素是否为具体整数，失败时抛出异常，附带文件/行号信息
- `asIntArrayRefSlowAlloc()` - 最慢方式，强制分配新缓冲区并逐个转换元素（通过 `guard_int()` 获取具体值）

**IntArrayRef → SymInt 方向：**
- `fromIntArrayRefUnchecked()` - 直接重新解释转换
- `fromIntArrayRefKnownNonNegative()` - 文档建议使用的语义化构造器（包装 unchecked 版本）
- `fromIntArrayRefSlow()` - 验证每个整数是否在 SymInt 的范围内

**SymIntArrayRef.cpp** 仅包含空的命名空间声明，实现全在头文件中（内联函数）。

### 关键点

- **设计目的** - 支持符号执行（symbolic execution），允许整数包含动态符号而非仅硬编码值
- **heap_allocated 标记** - SymInt 可能包含堆分配的大负整数，无法简单重新解释转换
- **宏定义** - `C10_AS_INTARRAYREF_SLOW` 和 `C10_AS_INTARRAYREF_SLOW_ALLOC` 自动捕获调用位置信息

---

### 功能总结

- 类型别名：SymIntArrayRef（符号整数数组引用）
- 单向转换：SymInt → int64_t（4种方式，安全性递增）
- 反向转换：int64_t → SymInt（3种方式，验证强度递增）
- 核心作用：在 PyTorch 的符号形状推理中桥接动态符号和具体整数值
