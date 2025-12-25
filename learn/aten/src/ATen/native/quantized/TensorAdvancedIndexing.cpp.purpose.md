这个文件实现了量化张量的高级索引操作，主要包括以下功能：

**核心功能：**

- **masked_fill_ 操作**：使用布尔掩码填充量化张量
  - CPU 版本：`masked_fill_impl_quantized_cpu` 支持标量和张量值
  - CUDA 版本：`masked_fill_impl_quantized_cuda` 支持标量和张量值
  - 验证掩码必须是布尔类型，支持命名张量推理

- **index_put_ 操作**：通过索引将值放入量化张量
  - CPU 版本：`_index_put_impl_quantized_cpu_`
  - CUDA 版本：`_index_put_impl_quantized_cuda_`
  - 支持多维索引和值广播
  - 验证值不能是量化的，仅支持 per-tensor affine 量化方案

**关键特性：**

- 仅支持 `c10::kPerTensorAffine` 量化方案（per-tensor 量化）
- 不支持累积模式（accumulate=False）
- 保留量化参数（q_scale 和 q_zero_point）
- 处理内存重叠警告和设备一致性检查
- CUDA 版本支持确定性算法（使用排序内核）
- 支持 masked_fill 的快速路径优化

**辅助函数：**

- `make_index_put_iterator`：构建张量迭代器用于索引操作
- 调度存根定义：`masked_fill_kernel_quantized_stub`、`index_put_kernel_quantized_stub`、`index_put_with_sort_quantized_stub`
