这个文件实现了 CUDA 中拉盖尔多项式（Laguerre polynomial）的前向计算核心。

**主要功能：**

- **定义 CUDA 核函数** - `laguerre_polynomial_l_kernel_cuda()` 处理拉盖尔多项式的 GPU 计算
- **双路径实现** - 通过 `AT_USE_JITERATOR()` 宏选择两种执行方式：
  - JIT 编译路径：使用 `opmath_jitted_gpu_kernel_with_scalars` 预编译的字符串内核
  - 标准路径：使用 Lambda 函数调用 `laguerre_polynomial_l_forward<scalar_t, true>()`
- **类型派发** - `AT_DISPATCH_FLOATING_TYPES` 宏支持所有浮点类型（float、double 等）
- **标量操作** - `gpu_kernel_with_scalars` 处理两个标量输入（x 和 n），返回计算结果
- **存根注册** - `REGISTER_DISPATCH` 宏将实现注册到调度系统，允许 CPU/GPU 代码统一调用

**核心特性：**

- 支持单精度和双精度浮点数
- GPU 批量并行处理
- 与 PyTorch 张量迭代器集成
- JIT 编译优化选项
