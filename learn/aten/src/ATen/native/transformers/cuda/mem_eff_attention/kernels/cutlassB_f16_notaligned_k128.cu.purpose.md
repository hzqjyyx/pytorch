这个文件定义了用于**内存高效注意力机制反向传播**的 CUDA 内核。

## 核心功能

该文件包含 5 个全局 CUDA 内核函数，针对不同的 GPU 架构（SM50/SM70/SM75）和不同的分块大小进行了优化：

1. **fmha_cutlassB_f16_notaligned_128x64_k128_sm70** - SM70 架构，128×64 分块
2. **fmha_cutlassB_f16_notaligned_128x64_k128_sm75** - SM75 架构，128×64 分块  
3. **fmha_cutlassB_f16_notaligned_64x64_k128_sm50** - SM50 架构，64×64 分块
4. **fmha_cutlassB_f16_notaligned_64x64_k128_sm70** - SM70 架构，64×64 分块
5. **fmha_cutlassB_f16_notaligned_64x64_k128_sm75** - SM75 架构，64×64 分块

每个内核都使用 `AttentionBackwardKernel` 模板类，以 **float16 (half_t)** 进行计算。

## 执行流程

- 内核检查当前编译的 CUDA 架构是否与目标架构匹配
- 通过 `advance_to_block()` 进行动态块调度
- 调用对应的 `attention_kernel()` 方法执行反向传播计算
- 不匹配的架构会输出致命错误信息

## 关键特性

- **auto-generated**：文件由 generate_kernels.py 自动生成
- **K 值固定为 128**：Head dimension 设定为 128
- **非对齐数据访问**：支持不对齐的张量访问模式
- **架构特定优化**：针对不同 GPU 代数的性能优化

## 简要总结

- 自动生成的反向传播 CUDA 内核文件
- 针对多个 GPU 架构（SM50/70/75）的优化实现
- 使用 CUTLASS 库框架进行高效矩阵计算
- 支持半精度（FP16）浮点运算
- 处理 FlashAttention 类型的内存高效注意力机制
