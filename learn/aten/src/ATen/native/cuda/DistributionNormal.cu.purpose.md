- **文件职责**：CUDA 实现的正态分布采样内核注册文件

- **核心函数 `normal_kernel`**：
  - 接收张量、均值(mean)、标准差(std) 和可选的随机数生成器
  - 获取 CUDA 生成器实例（若未提供则使用默认生成器）
  - 委托给模板实现 `templates::cuda::normal_kernel` 执行实际采样

- **关键依赖**：
  - `DistributionTemplates.h`：包含模板化的正态分布采样逻辑
  - `CUDAGeneratorImpl`：CUDA 随机数生成器实现

- **调度注册**：
  - 通过 `REGISTER_DISPATCH(normal_stub, &normal_kernel)` 将该内核注册到全局调度系统
  - 允许上层 API 通过 `normal_stub` 符号调用此 CUDA 实现

- **宏定义**：
  - `TORCH_ASSERT_NO_OPERATORS` 禁止操作符重载（保证编译隔离）
