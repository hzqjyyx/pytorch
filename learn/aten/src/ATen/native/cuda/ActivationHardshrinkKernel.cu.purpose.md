这个文件实现了 CUDA 版本的 Hardshrink 激活函数核心。

**主要功能：**

- **Hardshrink 激活函数实现**：根据阈值 λ 对输入张量进行硬收缩处理
  - 如果 |a| ≤ λ，输出为 0
  - 如果 |a| > λ，输出保持原值 a

- **类型支持**：处理浮点数类型（Float32、Float64）和半精度类型（Float16、BFloat16）

- **GPU 并行化**：使用 `gpu_kernel` 宏和 `GPU_LAMBDA` 在 CUDA 设备上并行处理张量元素

- **核心流程**：
  1. 接收 TensorIterator 和阈值参数
  2. 通过 `AT_DISPATCH_FLOATING_TYPES_AND2` 宏进行类型分发
  3. 将标量阈值转换为对应的数据类型
  4. 为每个张量元素应用 Hardshrink 操作

- **注册机制**：通过 `REGISTER_DISPATCH` 将实现注册到调度系统，供上层 API 调用
