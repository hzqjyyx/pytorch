## ActivationMishKernel.cu 文件分析

这个文件实现了 Mish 激活函数的 CUDA 核心计算。Mish 是一个平滑的非线性激活函数。

**Forward Pass (mish_kernel)**

计算公式：`output = x * tanh(log1p(exp(x)))`

- 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 宏支持多种浮点数类型（Float32、Float64、Float16、BFloat16）
- 通过 `gpu_kernel` 在 GPU 上并行执行计算
- 使用 `opmath_t` 类型进行中间计算以提高精度
- 调用 CUDA 兼容函数：`exp()`、`log1p()`、`tanh()` 计算最终结果

**关键设计点**

- `opmath_t = at::opmath_type<scalar_t>` 确保低精度输入（Half、BFloat16）使用高精度中间计算
- 使用 `log1p(exp(x))` 而非直接 `log(1+exp(x))` 以避免数值溢出
- `c10::cuda::compat::` 命名空间提供平台兼容的 CUDA 数学函数

**核心特性**

- bullet: GPU 并行化实现 Mish 激活函数正向传播
- bullet: 支持多种浮点数据类型（FP32、FP64、FP16、BF16）
- bullet: 使用数值稳定的计算方式（log1p）防止溢出
- bullet: 低精度输入通过高精度中间计算保证结果准确性
- bullet: 与 PyTorch 分发机制集成（REGISTER_DISPATCH）
