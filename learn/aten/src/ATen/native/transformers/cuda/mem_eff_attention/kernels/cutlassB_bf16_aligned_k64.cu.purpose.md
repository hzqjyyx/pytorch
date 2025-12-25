这个文件定义了两个 CUDA 全局函数，用于实现高效注意力机制的反向传播计算：

• **函数1**: `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80`
  - 针对序列对齐场景的反向传播内核
  - 使用 bfloat16 数据类型，块大小 64x64，头维度 k=64
  - 目标架构：SM80（Ada、Ampere 及更新）

• **函数2**: `fmha_cutlassB_bf16_aligned_64x64_k64_sm80`
  - 标准反向传播内核（非序列对齐版本）
  - 相同配置：bfloat16、64x64 块大小、k=64
  - 目标架构：SM80

• **核心逻辑**（两个函数类似）：
  - 检查 CUDA 计算能力是否在 SM80-SM100 范围内
  - 若满足条件，调用 `AttentionBackwardKernel` 的 `attention_kernel` 方法执行实际计算
  - 若架构不匹配，打印错误信息
  - 使用 `advance_to_block()` 处理多块网格分配

• **自动生成**：该文件由 `generate_kernels.py` 自动生成，不应手动编辑
