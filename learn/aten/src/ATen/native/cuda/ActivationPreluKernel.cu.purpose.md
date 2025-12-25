## ActivationPreluKernel.cu 文件分析

**PReLU (Parametric Rectified Linear Unit) 前向传播**
- `prelu_kernel` 函数实现了 PReLU 激活函数的 CUDA 核心计算
- 支持浮点数类型：float32、float64、bfloat16、float16
- 计算逻辑：`input > 0 ? input : weight * input`
  - 当输入为正时，直接输出输入值
  - 当输入为负时，乘以可学习的权重参数

**PReLU 反向传播**
- `prelu_backward_kernel` 计算梯度
- 返回两个梯度值：`grad_input` 和 `grad_weight`
- 梯度计算逻辑：
  - `grad_input = mask ? grad : weight * grad`（输入梯度）
  - `grad_weight = mask ? 0 : input * grad`（权重梯度）

**核心特性**
- 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 宏进行类型分发
- 利用 `gpu_kernel` 和 `gpu_kernel_multiple_outputs` 进行并行计算
- 使用 `REGISTER_DISPATCH` 注册分发表，将抽象实现与具体 CUDA 核心绑定

---

• PReLU 前向计算：条件激活函数，负值乘以可学习权重
• PReLU 反向计算：同时返回输入和权重的梯度
• 多精度支持：float32/64、bfloat16、float16
• CUDA 并行化：基于网格和循环的高性能并行实现
