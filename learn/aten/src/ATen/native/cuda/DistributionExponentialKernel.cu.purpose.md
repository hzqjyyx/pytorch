**DistributionExponentialKernel.cu 主要功能：**

- 实现 CUDA 上的指数分布（Exponential Distribution）随机数生成内核
- 通过 `exponential_kernel()` 函数接收 TensorIterator、lambda 参数和可选的随机数生成器
- 获取 CUDA 随机数生成器实例（若未提供则使用默认生成器）
- 调用模板化的 CUDA 指数分布内核实现（位于 DistributionTemplates.h）
- 通过 REGISTER_DISPATCH 宏将函数注册到 exponential_stub 分发器，供 PyTorch 框架调用
