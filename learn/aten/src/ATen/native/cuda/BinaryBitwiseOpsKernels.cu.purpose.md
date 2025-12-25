This file implements CUDA kernels for bitwise operations on tensors. Here's what it does:

**主要功能：**

- **BitwiseAndFunctor**: 实现按位与操作（`&`），对于布尔值使用逻辑与（`&&`）
- **BitwiseOrFunctor**: 实现按位或操作（`|`），对于布尔值使用逻辑或（`||`）
- **BitwiseXorFunctor**: 实现按位异或操作（`^`），对于布尔值使用不相等比较（`!=`）
- **bitwise_and_kernel_cuda**: 注册并执行按位与的 CUDA 内核
- **bitwise_or_kernel_cuda**: 注册并执行按位或的 CUDA 内核
- **bitwise_xor_kernel_cuda**: 注册并执行按位异或的 CUDA 内核
- **AT_DISPATCH_INTEGRAL_TYPES_AND**: 为整数和布尔类型分发不同的实现
- **opmath_symmetric_gpu_kernel_with_scalars**: 通用 GPU 内核包装器，处理张量迭代和标量操作
- **REGISTER_DISPATCH**: 将各个内核注册到调度系统

**核心设计：**
- 为布尔类型提供专门的处理（逻辑操作而非位操作）
- 支持整数类型的原生位操作
- 使用 CUDA 的 `__device__` 和 `__forceinline__` 优化性能
