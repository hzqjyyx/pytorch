# tags.yaml 文件功能分析

这是 PyTorch ATen 库中的标签定义文件，用于标记原生操作符（native operators）的特性和行为。

## 主要功能

- **inplace_view**: 标记仅修改张量元数据的操作符
- **pt2_compliant_tag**: 标记与 PyTorch 2.0 编译 API（torch.compile、torch.export）兼容的操作符
- **view_copy**: 标记视图操作符的 `_copy` 变体（如 `view_copy`）
- **dynamic_output_shape**: 标记输出形状取决于输入张量数据的操作符
- **data_dependent_output**: 标记输出（非张量）依赖于张量输入数据的操作符，无法在 meta tensor 或符号追踪中运行
- **generated**: 标记由代码生成器自动生成的操作符（不在 native_functions.yaml 中明确定义）
- **nondeterministic_seeded**: 标记随机操作符，结果由 Generator 控制
- **nondeterministic_bitwise**: 标记不保证按位等价性的操作符
- **needs_fixed_stride_order**: 标记要求特定步幅顺序的操作符（Inductor 编译时）
- **flexible_layout**: 标记支持灵活步幅/存储偏移的操作符
- **core**: 标记核心 ATen 操作符，经过分解和函数化后的子集，完全函数式且符合 SSA
- **pointwise**: 标记逐元素操作符，输出每个元素仅由对应的广播输入元素计算得出
- **maybe_aliasing_or_mutating**: 标记无法静态确定是否函数式的操作符
