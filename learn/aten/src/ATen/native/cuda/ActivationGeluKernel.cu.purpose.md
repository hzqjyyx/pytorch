这个文件实现了 CUDA 上的 GELU（Gaussian Error Linear Unit）激活函数及其反向传播。

**主要功能：**

- **GeluCUDAKernelImpl** - GELU 前向计算
  - 支持两种近似方式：Tanh 和 Erf
  - Tanh 版本：使用 `0.5 * x * (1 + tanh(β * (x + κ * x³)))` 公式
  - Erf 版本：使用 `0.5 * x * (1 + erf(x / √2))` 公式
  - 支持 Float32、Float16、BFloat16 数据类型

- **GeluBackwardCUDAKernelImpl** - GELU 反向传播计算
  - 同样支持 Tanh 和 Erf 两种模式
  - Tanh 模式：计算导数涉及 tanh 的链式法则
  - Erf 模式：计算 CDF（累积分布函数）和 PDF（概率密度函数）的组合导数

- **核心实现细节**
  - 使用 GPU kernel 进行元素级并行计算
  - 通过 `AT_DISPATCH_FLOATING_TYPES_AND2` 宏处理多种浮点类型
  - 使用 opmath 类型以提高计算精度
  - 利用 thrust 和 CUDA Math 兼容库进行数学运算
