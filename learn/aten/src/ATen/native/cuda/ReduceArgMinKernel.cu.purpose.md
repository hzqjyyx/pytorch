• **文件目的**: 实现 CUDA 版本的 argmin 操作核心算法，用于找到张量中最小值的索引

• **核心函数**: 
  - `argmin_kernel_cuda_impl`: 模板函数，使用 `gpu_reduce_kernel` 和 `ArgMinOps` 执行归约操作
  - `argmin_kernel_cuda`: 包装函数，处理不同数据类型的分发

• **数据类型处理**:
  - 对 float16 (Half) 和 bfloat16 转换为 float 执行计算（避免实现 is_nan 和 warp_shfl_down）
  - 其他类型通过 `AT_DISPATCH_ALL_TYPES` 宏分发

• **关键操作**:
  - 使用 `ArgMinOps` 比较逻辑找最小值
  - 初始化为 `(upper_bound, 0)` - 最大值和索引 0
  - 返回 `thrust::pair<acc_t, int64_t>` - 最小值和其索引

• **注册**: 通过 `REGISTER_DISPATCH(argmin_stub, &argmin_kernel_cuda)` 注册到分发系统
