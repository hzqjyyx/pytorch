## ConjugateFallback.cpp 文件分析

这个文件实现了 PyTorch 中共轭（Conjugate）操作的回退机制。

**核心结构：**

1. **ConjFallback 类** (第5-10行)
   - 继承自 `MathOpFallback`
   - 注册到 `Conjugate` dispatch key
   - 通过 `is_bit_set()` 检查张量是否设置了共轭标志位

2. **conjugateFallback 函数** (第12-15行)
   - 创建 ConjFallback 对象并调用其 `fallback_impl()` 方法
   - 处理实际的共轭操作分发

3. **TORCH_LIBRARY_IMPL 注册** (第17-63行)
   - 第一个块：为所有操作注册通用回退处理
   - 第二个块：为特定操作注册穿透（fallthrough）实现

**主要功能：**

- **共轭位检查**：识别张量是否标记为共轭
- **操作穿透**：对于存储、复制、克隆等基础操作，直接穿透到下一个 dispatch key
- **线性代数支持**：dot、vdot、mm、bmm、addmm、baddbmm、linalg_svd 等操作的共轭处理
- **视图操作**：通过宏 `TORCH_VIEW_FNS` 和 `TENSOR_UTILITIES_AND_CONSTRUCTORS` 注册视图相关操作

**关键点：**

- 使用 dispatch 机制实现共轭操作的透明处理
- 对于不需要特殊处理的操作使用 fallthrough，避免不必要的开销
- 支持复数张量的共轭操作而无需物理复制数据
