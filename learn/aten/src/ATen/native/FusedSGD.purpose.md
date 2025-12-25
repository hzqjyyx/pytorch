**FusedSGD 文件功能分析**

这两个文件实现了 PyTorch 中融合的 SGD（随机梯度下降）优化器的 CPU 核心逻辑。

**FusedSGD.h** 定义了核心接口：
- `fused_sgd_fn` 函数指针类型，定义了单个参数张量的 SGD 更新签名
- `fused_sgd_stub` dispatch stub，用于在不同硬件后端（CPU/CUDA）间切换实现

**FusedSGD.cpp** 实现了两个 CPU 内核函数：
1. 第一个 `_fused_sgd_kernel_cpu_`（标量学习率版本，line 18-61）：
   - 接收参数列表、梯度列表、动量缓冲区列表
   - 处理可选的梯度缩放和 inf 检测（用于混合精度训练）
   - 验证输入张量的一致性
   - 对每个参数调用 `fused_sgd_stub` 进行实际的 SGD 更新

2. 第二个 `_fused_sgd_kernel_cpu_`（张量学习率版本，line 63-81）：
   - 接收张量形式的学习率
   - 将其转换为标量后调用第一个版本

**主要特性：**

- 支持动量（momentum）和 Nesterov 加速
- 支持权重衰减（weight decay）
- 支持最大化模式（maximize）
- 处理首步特殊逻辑（is_first_step）
- 支持梯度缩放和无穷值检测（混合精度训练）
- 使用 dispatch stub 实现后端无关性

**核心流程：**

- 验证 inf 标志→检查输入张量数量一致性→处理动量缓冲→循环调用底层 stub 更新每个参数
