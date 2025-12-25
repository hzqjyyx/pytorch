这个文件实现了 CUDA 上的融合 Adam 优化器（带 AMSGrad 变体）的核心逻辑。

**主要功能：**

- **两个重载函数** `_fused_adam_amsgrad_cuda_impl_()`，分别处理标量学习率和张量学习率
- **参数整合**：将优化器所需的所有张量（参数、梯度、一阶矩估计、二阶矩估计、最大二阶矩）组织成向量列表传递给多张量应用函数
- **指针提取**：从可选张量（梯度缩放因子、无穷大检测）中提取 float 指针，用于混合精度训练和梯度溢出检测
- **类型分发**：使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 宏支持 float32、float64、float16、bfloat16 等数据类型
- **融合核函数调用**：调用 `multi_tensor_apply_for_fused_optimizer` 并传入 `FusedAdamMathFunctor` 以在单个 CUDA 核函数中完成参数更新，避免多次内存访问
- **优化配置**：支持 weight decay、epsilon 值、maximize 模式（极大化而非最小化）等 Adam 优化器参数

**关键点：**

- 融合优化通过减少全局内存往返来提升性能
- AMSGrad 变体通过追踪二阶矩的最大值来改进收敛稳定性
- 支持混合精度训练和梯度缩放机制
