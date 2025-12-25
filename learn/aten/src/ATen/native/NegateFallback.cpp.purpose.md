## NegateFallback.cpp 功能分析

**核心功能：**
- 实现了 `NegFallback` 结构体，继承自 `MathOpFallback`，用于处理张量的负数位（neg bit）标记
- 当张量被标记为负数视图时，通过 fallback 机制处理相关操作

**工作流程：**
1. `NegFallback` 构造函数注册 `DispatchKey::Negative` 分发键，关联"negation"操作
2. `is_bit_set()` 方法检查张量是否设置了负数位标记（`tensor.is_neg()`）
3. `negationFallback()` 函数在调用时创建 `NegFallback` 对象并执行 fallback 实现

**分发规则注册：**
- 通过 `TORCH_LIBRARY_IMPL` 宏注册两个分发库：
  - `_` 命名空间：设置全局 fallback 处理器
  - `aten` 命名空间：为特定操作注册 fallthrough 行为

**Fallthrough 操作列表：**
- 存储/拷贝操作：`set_`、`copy_`、`clone`
- 取反操作：`neg_`、`resolve_neg`
- 共轭解析：`resolve_conj`
- 重复操作：`repeat_interleave`（3 个变体）
- 元数据检查：`_has_same_storage_numel`、`_new_zeros_with_same_feature_meta`
- 线性代数：`linalg_solve_triangular`、`linalg_svd`（及其输出变体）
- 视图函数和张量构造函数（通过宏展开）

**关键特性：**
• 负数位优化机制 - 延迟真实取反计算，仅标记张量状态
• Fallback 设计 - 对不支持负数位的操作进行降级处理
• Fallthrough 注册 - 允许特定操作跳过负数位处理逻辑
• 支持链式操作 - 负数位可在多个操作间传播
