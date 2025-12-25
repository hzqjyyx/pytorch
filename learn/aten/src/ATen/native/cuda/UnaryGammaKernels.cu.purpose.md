## UnaryGammaKernels.cu 文件分析

这个文件实现了 PyTorch ATen 库中三个 gamma 函数相关的 CUDA 核函数：

**digamma_kernel_cuda** (行 19-40)
- 计算 digamma 函数（gamma 函数的对数导数）
- 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持 Float、Double、Half 和 BFloat16 数据类型
- 条件编译：若启用 JITERATOR，使用 JIT 编译的 GPU 核函数；否则使用 `calc_digamma` 函数

**trigamma_kernel_cuda** (行 43-65)
- 计算 trigamma 函数（digamma 函数的导数）
- 实现结构与 digamma 类似，调用 `calc_trigamma`

**polygamma_kernel_cuda** (行 67-102)
- 计算多阶 polygamma 函数（第 n 阶导数）
- 特殊优化：n=0 时直接调用 digamma，n=1 时调用 trigamma
- 对于 n≥2，通过 `calc_polygamma` 计算，传递 n 作为额外参数

**关键特性：**

- **双执行路径**：JITERATOR 模式（JIT 编译）vs 传统 GPU 核函数
- **多精度支持**：Float32、Float64、Float16、BFloat16
- **TensorIterator**：使用 PyTorch 的张量迭代器处理多维数据
- **函数注册**：通过 `REGISTER_DISPATCH` 将实现绑定到分发机制

**核心功能总结：**

- 为 PyTorch 提供高效的 CUDA gamma 函数计算
- 支持多种浮点数据类型和张量形状
- 与 JIT 编译基础设施集成以获得最佳性能
