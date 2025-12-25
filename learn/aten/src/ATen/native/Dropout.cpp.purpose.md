## Dropout.cpp 文件功能分析

**核心目的**：实现 PyTorch 中的 Dropout 相关操作的 CPU 端实现。

**主要函数**：

- **`make_feature_noise()`** (30-41行)：为 Feature Dropout 生成噪声张量，保留前两个维度，其余维度设为1

- **`is_fused_kernel_acceptable()`** (43-45行)：判断是否可使用融合内核（CUDA/XPU/Lazy/PrivateUseOne 设备，且 0 < p < 1）

- **`_dropout_impl<>()`** (61-89行)：通用 Dropout 实现函数
  - 处理边界情况（p=0、p=1、非训练模式）
  - 使用伯努利分布生成掩码
  - 支持四种模式：普通/特征 × 普通/alpha dropout

- **`native_dropout_cpu()`** (104-125行)：CPU 实现的 native_dropout，返回输出张量和掩码，支持训练/推理模式切换

- **`native_dropout_backward()`** (127-130行)：反向传播计算梯度

- **公开接口**（132-173行）：
  - `dropout()` / `dropout_()`
  - `feature_dropout()` / `feature_dropout_()`
  - `alpha_dropout()` / `alpha_dropout_()`
  - `feature_alpha_dropout()` / `feature_alpha_dropout_()`

**关键机制**：

• 使用 Bernoulli 分布生成随机掩码
• 根据保留概率 (1-p) 缩放输出以保持期望值
• Alpha Dropout 额外调整以保持自归一化特性（针对 SELU 激活函数）
• 宏 `ALIAS_SPECIALIZATION` 简化四种 Dropout 变体的实现
• 支持原地操作（带 `_` 后缀）和非原地操作
• 嵌套张量或可用融合内核时调用 `native_dropout()`，否则使用通用实现
