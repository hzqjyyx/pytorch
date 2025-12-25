**ReduceAMinMaxKernel.cu 文件功能**

这个文件实现了 PyTorch 在 CUDA 上计算张量最小值和最大值的内核函数。

**核心功能：**

- **_min_max_values_kernel_cuda_impl**（第20-28行）：通用的 min-max 归约实现，调用 `gpu_reduce_kernel` 使用 `MinMaxOps` 算子处理数据。初始化最大值为类型的上界，最小值为类型的下界。

- **aminmax_allreduce_launch_kernel**（第30-35行）：全局归约版本，对整个张量计算 min-max，支持所有数值类型（包括 bfloat16、half、bool）。

- **aminmax_launch_kernel**（第37-47行）：维度归约版本，沿指定维度计算 min-max。

**关键特点：**

- 使用 `AT_DISPATCH_ALL_TYPES_AND3` 宏实现对多种数据类型的支持
- 利用 thrust::pair 同时返回最小值和最大值
- 依赖 `TensorIterator` 处理复杂的张量布局和维度逻辑
- 调用统一的 GPU 归约内核框架 (`gpu_reduce_kernel`)

**汇总：**

- 计算张量沿维度的最小/最大值对
- 支持全局和逐维度两种归约模式
- 支持所有主要数据类型（整数、浮点、bool、半精度）
- 代码简洁，核心逻辑由共享的 GPU 归约框架提供
