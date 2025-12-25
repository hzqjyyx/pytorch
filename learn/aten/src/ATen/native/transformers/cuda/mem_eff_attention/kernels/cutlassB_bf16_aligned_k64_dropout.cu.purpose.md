这是一个 CUDA 核函数文件，由自动生成脚本（generate_kernels.py）创建。

**主要功能：**

- 定义了针对 SM80 架构（H100/A100 GPU）的反向传播（backward）FMHA（Flash Attention）核函数
- 核函数名称：`fmha_cutlassB_bf16_aligned_64x64_k64_dropout_sm80`
- 使用 Cutlass 库实现高效的注意力机制计算
- 支持 bfloat16 数据类型和 dropout 正则化
- 处理对齐的 64×64 矩阵操作，密钥维度（k）为 64
- 包含架构检查：只在 SM80-SM120（Ampere 到 Ada）上正确运行，其他架构会输出致命错误信息
- 实现内存高效的注意力计算，通过分块处理和优化内存访问模式来提高性能
