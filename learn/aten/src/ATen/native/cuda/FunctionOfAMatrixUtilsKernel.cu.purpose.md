这个文件实现了 CUDA 核心函数，用于矩阵函数运算中的线性组合计算。

**主要功能：**

• **`_elemwise_kernel`** - 通用的逐元素 CUDA 核函数，使用配置的线程数和每线程处理元素数来执行任意函数 `f`

• **`_lauch_kernel`** - 核函数启动器，负责计算 Grid/Block 维度、获取当前 CUDA 流、调用核函数并检查启动错误

• **`_compute_linear_combination_internal_kernel`** - 核心计算函数，执行线性组合求和：`out += in * coeff`，支持多个求和项迭代，处理 32 位寻址限制的分片执行

• **`_compute_linear_combination_cuda_kernel`** - 分派函数，支持所有数据类型（包括复数、半精度、BFloat16 等），调用内部核函数

• **`REGISTER_DISPATCH`** - 注册分派表，将 CPU 端接口与 CUDA 实现绑定

**核心工作流：** 接收输出、输入、系数三个张量，对每个元素执行累加求和，支持多个求和项的线性组合操作。
