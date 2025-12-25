这个文件实现了 CUDA 上的 Hermite 多项式 H（physicists' Hermite）的前向计算：

- **文件用途**：为 PyTorch 张量操作库提供 GPU 加速的 Hermite 多项式 H 计算
- **核心函数**：`hermite_polynomial_h_kernel_cuda()` - 在 CUDA 中执行多项式计算
- **双路径实现**：
  - `AT_USE_JITERATOR()` 启用时：使用 JIT 编译器生成优化的 GPU 代码（`opmath_jitted_gpu_kernel_with_scalars`）
  - 否则：使用编译时定义的 Lambda 函数执行计算（`gpu_kernel_with_scalars`）
- **计算内容**：调用 `hermite_polynomial_h_forward<scalar_t, true>(x, n)` 计算给定 x 和阶数 n 的 Hermite H 多项式值
- **类型支持**：仅处理浮点数类型（通过 `AT_DISPATCH_FLOATING_TYPES` 宏）
- **注册机制**：通过 `REGISTER_DISPATCH` 将此 CUDA 实现注册到对应的分发存根（stub）
