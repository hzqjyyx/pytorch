这个文件定义了用于高效注意力机制反向传播的 CUDA 内核。

**文件结构：**

该文件是自动生成的（根据注释 "This file is auto-generated. See 'generate_kernels.py'"），包含 5 个全局 CUDA 内核函数，分别针对不同的 GPU 架构优化：

**内核函数列表：**

- `fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm50` - SM50 架构，128x64 线程块，65536 大小 K，带 dropout
- `fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm70` - SM70 架构，128x64 线程块配置
- `fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm75` - SM75 架构，128x64 线程块配置
- `fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm70` - SM70 架构，64x64 线程块配置
- `fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm75` - SM75 架构，64x64 线程块配置

**关键特征：**

- 使用 `cutlass::half_t`（FP16 半精度浮点）进行计算
- `dropout=true` - 支持在反向传播中应用 dropout
- 内核通过 `AttentionBackwardKernel` 模板类实现
- 包含运行时架构检查（`__CUDA_ARCH__`），确保内核仅在目标 GPU 架构上执行
- 使用 `__launch_bounds__` 指令优化线程调度

**核心功能：**

- 实现 Flash Attention 等高效注意力机制的反向传播计算
- 针对不同 GPU 架构（Maxwell, Pascal, Volta）的性能优化
- 支持大序列长度（K=65536）和混合精度计算
