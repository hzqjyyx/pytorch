## WeightNorm.cpp 核心功能分析

**主要作用**：实现权重归一化（Weight Normalization）的前向和反向计算

**关键函数**：

- **`norm_except_dim()`** (30-48行)
  - 计算张量沿特定维度外的范数
  - 处理三种特殊情况（dim=-1, 0, last_dim）优化性能
  - 其他维度通过转置后递归调用

- **`weight_norm_cpu()`** (50-63行)
  - CPU上的权重归一化前向计算
  - 输入：v（原始权重）、g（缩放因子）、dim（作用维度）
  - 输出：w（归一化后的权重）、norm（计算的范数）
  - 处理BFloat16特殊情况（范数保持Float精度）

- **`weight_norm_backward_cpu()`** (65-80行)
  - CPU上的反向传播
  - 接收梯度和保存的中间值，计算grad_v和grad_g

- **`_weight_norm()`** (82-110行)
  - 决策函数：选择融合实现或原始实现
  - 融合路径：调用`_weight_norm_interface`（针对dim=0或last_dim）
  - 原始路径：使用`norm_except_dim`的可微实现

- **`_weight_norm_differentiable_backward()`** (116-160行)
  - 可微反向路径（用于高阶导数）
  - 分别处理dim=0和dim=last_dim两种情况
  - 使用解析梯度公式进行计算

**核心特点**：

- 支持可微操作链，可用于高阶导数
- 优化了特定维度（首尾维度）的计算性能
- 通过动态调度选择最优实现路径

**输出总结**：

• 实现权重归一化的CPU前向和反向计算
• 提供融合优化路径和通用可微路径两种实现
• `norm_except_dim()`处理沿特定维度外的范数计算
• 支持自动求导和高阶导数
• 特殊处理低精度类型（Half、BFloat16）
