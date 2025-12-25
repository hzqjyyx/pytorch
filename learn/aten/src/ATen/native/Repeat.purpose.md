## Repeat.h - 通用模板

定义了 `repeat_interleave_common` 模板函数，用于处理 repeat_interleave 操作的核心逻辑：

- 接收一个 1D repeat 张量，指定每个元素要重复的次数
- 通过 cumsum 计算累积和来确定结果位置
- 分配一个新张量来存储重复后的结果
- 调用传入的 `compute` 函数来填充结果数据
- 支持可选的 output_size 参数来预指定输出大小

## Repeat.cpp - 具体实现和高级功能

**CPU 实现：**
- `compute_cpu` 函数并行填充结果张量
- 利用 repeat 和 cumsum 数据计算每个元素的起止位置
- 使用 `at::parallel_for` 实现多线程并行化

**多维 repeat_interleave（Tensor 版本）：**
- `repeat_interleave_symint(Tensor repeats)` 支持沿指定维度重复
- 处理 0-dim 和 1-dim repeat 张量的不同情况
- 保留和恢复张量的共轭（conj）和负（neg）标志位
- 使用 `index_select` 实现最终的重复操作

**标量 repeat_interleave（SymInt 版本）：**
- `repeat_interleave_symint(SymInt repeats)` 支持单个重复次数
- 通过 unsqueeze → expand → flatten 的流程实现重复
- 支持符号整数（SymInt）用于动态形状跟踪

---

**核心功能汇总：**
- 提供 repeat_interleave 操作的 CPU 后端实现
- 支持按张量或标量指定重复次数
- 支持多维张量沿指定维度重复
- 使用符号整数处理动态形状
- 并行化 CPU 计算以提高性能
- 保留张量的共轭和负号属性
