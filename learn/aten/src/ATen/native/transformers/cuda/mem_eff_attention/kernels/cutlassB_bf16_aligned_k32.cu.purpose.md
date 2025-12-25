**文件功能分析**

这是一个 CUDA 核函数文件，实现高效attention机制的反向传播计算。

- **自动生成文件**：由 `generate_kernels.py` 脚本生成，文件头明确标注
- **核函数定义**：定义两个全局CUDA核函数
  - `fmha_cutlassB_bf16_aligned_64x64_k32_seqaligned_sm80`：支持序列对齐模式
  - `fmha_cutlassB_bf16_aligned_64x64_k32_sm80`：标准版本
- **数据类型**：使用 `bfloat16_t`（16位脑浮点数）进行计算
- **硬件目标**：针对NVIDIA SM80-SM100（A100等GPU架构）优化
- **CUTLASS库**：基于Meta的CUTLASS模板库实现，提供高性能矩阵运算
- **参数配置**：
  - Block尺寸：64×64
  - Head维度(k)：32
  - 对齐要求：内存对齐优化
- **条件编译**：
  - 检查GPU架构是否符合SM80-SM100范围
  - 若编译架构不匹配则打印错误信息
  - 支持动态块分配（`advance_to_block()`）
- **核函数调用链**：`AttentionBackwardKernel::attention_kernel()` 执行实际的反向传播计算
