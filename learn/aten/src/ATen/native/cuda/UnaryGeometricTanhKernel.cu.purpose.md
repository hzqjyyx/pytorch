这个文件实现了 PyTorch 中 CUDA 上的 `tanh`（双曲正切）函数核心。

**主要功能：**

- 定义 `tanh_kernel_cuda()` 函数，处理张量元素的 tanh 运算
- 区分两种数据类型路径：
  - **复数类型**（complex）：使用 `std::tanh` 并通过 opmath_t 转换处理 ComplexHalf
  - **浮点类型**（float, double, half, bfloat16）：直接调用 `::tanh`
- 根据 `AT_USE_JITERATOR()` 编译开关选择执行方式：
  - 启用时：使用 JIT 编译器动态生成 GPU 核心代码
  - 禁用时：使用预编译的 GPU Lambda 函数
- 通过 `TensorIterator` 自动处理多维张量的遍历和内存布局
- 注册分发机制将 `tanh_stub` 绑定到 `tanh_kernel_cuda` 实现

**关键点：**

- 支持复数和浮点两大类数据类型
- 使用 GPU Lambda 实现高性能并行计算
- JIT/预编译两套路径确保兼容性和性能
