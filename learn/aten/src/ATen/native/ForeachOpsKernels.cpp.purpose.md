# ForeachOpsKernels.cpp 主要功能分析

这个文件提供了 PyTorch 中批量张量操作（foreach operations）的**慢速回退实现**（slow fallback）。当没有优化的 CUDA/CPU kernel 可用时，使用这些实现。

## 核心设计模式

文件通过**宏定义**批量生成函数，避免重复代码。主要有以下几类宏：

### 1. FOREACH_BINARY_OP_TENSOR (lines 68-103)
- 生成 tensor 与单个 scalar tensor 的二元操作
- 包含原地版本 (`_`) 和返回新 tensor 版本
- 验证 scalar tensor 必须是 0 维且只有 1 个元素
- 示例：`foreach_tensor_mul_tensor_kernel_slow_`

### 2. FOREACH_BINARY_OP_SCALAR (lines 142-163)
- 生成 tensor list 与单个 scalar 值的二元操作
- 对每个 tensor 执行相同的 scalar 操作
- 示例：用于 add/sub/mul/div/clamp_min/clamp_max/pow

### 3. FOREACH_BINARY_OP_SCALARLIST (lines 165-185)
- tensor list 与 scalar list 的逐元素二元操作
- 第 i 个 tensor 与第 i 个 scalar 配对
- 确保两个 list 大小匹配

### 4. FOREACH_BINARY_OP_LIST (lines 187-208)
- 两个 tensor list 的逐元素二元操作
- 第 i 个 tensor 与另一个 list 的第 i 个 tensor 操作

### 5. FOREACH_BINARY_OP_*_ALPHA (lines 105-140, 210-231)
- 带 alpha 参数的二元操作版本
- 用于 `add(other, alpha)` 这类需要缩放因子的操作

### 6. FOREACH_UNARY_OP (lines 233-252)
- 单目操作（如 sqrt, exp, abs, sin, cos 等）
- 对 list 中每个 tensor 独立执行

### 7. FOREACH_POINTWISE_OP_* (lines 254-331)
- 三参数逐点操作（如 addcdiv, addcmul）
- 公式：`input[i].op(tensors1[i], tensors2[i], scalar/scalars[i])`
- 支持单个 scalar、scalar list、或从 tensor 转换的 scalar list

## 实际生成的操作

**二元操作 (lines 333-361):**
- add, sub, mul, div (支持 list/scalar/scalarlist/tensor 变体)
- lerp (线性插值)
- clamp_min, clamp_max, pow

**单目操作 (lines 374-403):**
数学函数：sqrt, exp, log, log10, log1p, log2, abs, reciprocal, rsqrt
三角函数：sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
其他：ceil, floor, round, trunc, frac, erf, erfc, expm1, neg, lgamma, sigmoid, sign

**特殊操作:**
- `foreach_tensor_copy_list_kernel_slow_` (lines 363-372): 批量复制，支持 non_blocking
- `foreach_tensor_zero_slow_` (lines 458-464): 批量置零
- `foreach_tensor_norm_slow` (lines 466-476): 批量计算向量范数
- `foreach_tensor_max_slow` (lines 478-485): 批量求最大值
- `foreach_scalar_pow_list_kernel_slow` (lines 487-497): scalar 的 tensor list 次幂

**三元操作:**
- `foreach_tensor_ternary_lerp_slow` (lines 414-434): 三 tensor list 的 lerp
- addcdiv, addcmul (lines 405-412): `input + value * tensor1 / tensor2` 或 `* tensor1 * tensor2`

## 实现特点

1. **统一的 API 检查**: 所有函数开头调用 `check_foreach_api_restrictions()` 验证输入合法性
2. **命名约定**: 
   - `_slow` 后缀表示未优化的回退实现
   - `_` 后缀表示原地操作（in-place）
3. **内存预分配**: 非原地版本使用 `result.reserve()` 提前分配内存
4. **条件编译**: `AT_PER_OPERATOR_HEADERS` 控制头文件包含方式 (lines 7-64)

ROCm 和 Backward 相关内容：
- 本文件不包含 ROCm 特定代码
- 无 backward/梯度计算相关实现（这些在其他文件中）
