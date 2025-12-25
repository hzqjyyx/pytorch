这个文件实现了CUTLASS库的Epilogue操作，用于高效注意力机制的输出重新缩放。

**主要功能：**

- **MemoryEfficientAttentionNormalize类** (76-190行)：线程级别的操作符，对注意力计算的累积结果进行线性组合和归一化
  - 接收`s_prime`（行和）和`m_prime`（行最大值）作为缩放参数
  - 支持两个操作符重载：一个处理首次计算（只需累积值），一个处理后续迭代（需要源数据和累积值混合）

- **缩放公式**：
  - `alpha = isLast ? (1/s_prime[row]) : 1`（根据是否最后一次迭代决定是否归一化）
  - `beta = alpha * m_prime[row]`（当最大值变化时重新缩放源数据）
  - 输出 = `alpha * accumulator + beta * source`

- **数据类型转换**：使用NumericArrayConverter进行多步转换（源→计算类型→输出类型），支持浮点舍入策略

- **ApplyEpilogueOp特化** (195-230行)：为threadblock级别提供适配器，将row_id作为第一个参数传递给thread级操作符

- **主要特性**：
  - 处理被完全mask的行（行和为0），设置为1避免NaN
  - 支持isFirst/isLast标志控制是否需要源数据
  - 可配置的数据类型、累积类型和计算精度
