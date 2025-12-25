这个文件实现了 AdamW 优化器的 CUDA 融合内核（Fused AdamW Kernel）。

**主要功能：**

- 提供两个重载的 `_fused_adamw_kernel_cuda_()` 函数，分别处理标量学习率和张量学习率的情况
- 第一个重载（标量 lr）：接收学习率作为 double 类型参数，直接调用实现函数
- 第二个重载（张量 lr）：接收学习率作为 Tensor 类型参数，支持动态学习率调度
  - 若 lr 在 CPU 上，提取标量值后调用第一个重载
  - 若 lr 在 GPU 上，进行设备一致性检查，然后调用实现函数
- 支持两种优化器变体：
  - **AMSGrad 版本**：调用 `_fused_adamw_amsgrad_cuda_impl_()`，使用最大指数移动平均
  - **标准 AdamW 版本**：调用 `_fused_adamw_cuda_impl_()`
- 进行参数验证：检查所有张量具有相同的数据类型、设备和内存布局
- 验证可选参数（grad_scale、found_inf）与参数张量在同一 GPU 设备上
- 支持梯度缩放和无穷值检测用于混合精度训练

**关键参数：**
- params, grads, exp_avgs, exp_avg_sqs：模型参数、梯度、一阶和二阶矩估计
- max_exp_avg_sqs：用于 AMSGrad 的最大二阶矩
- state_steps：优化器步数
- lr, beta1, beta2, weight_decay, eps：超参数
- amsgrad, maximize：算法选项开关
