这个文件实现了 CUDA 上的 Threshold 激活函数内核。

**主要功能：**

- **Threshold 操作**：对输入张量中的每个元素进行阈值判断
  - 如果元素 ≤ 阈值，输出指定的替换值
  - 如果元素 > 阈值，输出原始元素值

- **GPU 核心实现** (`threshold_kernel_impl`)：
  - 使用 `gpu_kernel_with_scalars` 在 GPU 上并行处理
  - Lambda 函数定义了每个元素的计算逻辑

- **类型支持**：
  - 支持所有标准数值类型（int、float 等）
  - 额外支持 Half（float16）和 BFloat16

- **张量迭代**：
  - 使用 `TensorIteratorBase` 高效遍历张量元素
  - 通过 `AT_DISPATCH_ALL_TYPES_AND2` 宏进行类型分发

- **接口注册**：
  - `REGISTER_DISPATCH` 将 CUDA 实现注册到 `threshold_stub` 调度系统
