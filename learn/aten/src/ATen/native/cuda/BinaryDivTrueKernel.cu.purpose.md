## BinaryDivTrueKernel.cu 主要功能分析

这个文件实现了 CUDA 上的真实除法（true division）操作，即浮点数除法。

**核心逻辑：**

1. **入口函数** `div_true_kernel_cuda()` - 根据数据类型和操作数特征选择不同的实现路径

2. **ComplexHalf 特殊处理**（第22-34行）
   - 使用 JIT 编译器生成除法核函数，或回退到 opmath 版本
   - 处理复数半精度浮点数

3. **优化路径 - CPU 标量优化**（第35-48行）
   - 当第二个操作数是 CPU 标量时，将除法转换为乘以倒数
   - `a / b` → `a * (1/b)`
   - 权衡：精度损失一位，但计算效率更高

4. **通用路径**（第49-55行）
   - 支持浮点数和复数类型（包括 Half、BFloat16）
   - 使用 `DivFunctor` 直接执行元素级除法

5. **分发注册**（第59行）
   - 将实现注册到全局分发系统，供上层调用

**关键特点：**

- 使用 `TensorIterator` 进行迭代优化
- 支持 JIT 编译和预编译两种模式
- 针对常见场景（CPU 标量）的专门优化
- 处理精度类型转换（opmath_type）

**主要特性列表：**

- Floating-point 和 complex number 支持
- CPU 标量除法优化（倒数乘法）
- JIT 和原生 CUDA 核函数双路径
- 精度类型自动处理（Half、BFloat16、float、double、complex）
- TensorIterator 驱动的高效并行计算
