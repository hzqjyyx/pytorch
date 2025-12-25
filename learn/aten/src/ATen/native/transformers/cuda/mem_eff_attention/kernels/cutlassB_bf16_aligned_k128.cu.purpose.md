这个文件定义了四个CUDA全局函数，用于执行**注意力机制的反向传播**计算。这些是自动生成的内核，针对NVIDIA GPU的SM80架构（如A100）优化。

**核心函数：**

- `fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80` - 128x128配置，序列对齐版本
- `fmha_cutlassB_bf16_aligned_128x128_k128_sm80` - 128x128配置，标准版本
- `fmha_cutlassB_bf16_aligned_64x64_k128_seqaligned_sm80` - 64x64配置，序列对齐版本
- `fmha_cutlassB_bf16_aligned_64x64_k128_sm80` - 64x64配置，标准版本

**关键特性：**

- **数据类型**: bfloat16（脑浮点，用于深度学习）
- **架构约束**: `__CUDA_ARCH__ >= 800 && <= 1200`（SM80-SM100）
- **线程块启动边界**: 通过`__launch_bounds__`指定最大线程数和最小块数，优化GPU利用率
- **块级工作分配**: `p.advance_to_block()`确保动态分配工作给活跃的线程块
- **实际计算**: 调用`AttentionBackwardKernel::attention_kernel(p)`执行梯度计算
- **错误检查**: 若在不支持的GPU上编译，打印致命错误信息

**工作流程：**

1. 检查当前GPU计算能力是否在SM80-SM100范围内
2. 若符合，进行动态块调度检查
3. 执行注意力反向传播内核计算Q/K/V的梯度
4. 若不符合，输出硬件不兼容警告
