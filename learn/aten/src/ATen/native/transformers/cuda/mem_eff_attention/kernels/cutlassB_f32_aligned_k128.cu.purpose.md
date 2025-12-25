这个文件是 PyTorch 的高效注意力机制(Efficient Attention)CUDA 内核实现。

文件包含 5 个全局 CUDA 内核函数，每个针对不同的 GPU 架构(SM)版本：

**内核函数列表:**

- `fmha_cutlassB_f32_aligned_128x64_k128_sm80` - SM80-SM100 架构，block 尺寸 128x64
- `fmha_cutlassB_f32_aligned_64x64_k128_sm50` - SM50-SM70 架构，block 尺寸 64x64
- `fmha_cutlassB_f32_aligned_64x64_k128_sm70` - SM70-SM75 架构，block 尺寸 64x64
- `fmha_cutlassB_f32_aligned_64x64_k128_sm75` - SM75-SM80 架构，block 尺寸 64x64
- `fmha_cutlassB_f32_aligned_64x64_k128_sm80` - SM80-SM100 架构，block 尺寸 64x64

**核心特点:**

- 使用 CUTLASS 库实现高效矩阵运算
- 数据类型为 float32
- 所有内核都是对齐的(aligned)，key 维度为 128
- 使用 `AttentionBackwardKernel` 模板类处理注意力反向传播计算
- 每个内核包含架构检查，确保运行在正确的 GPU 架构上，否则打印错误信息
- 自动生成文件(见第 8 行注释)

**主要功能:**

- 实现多层注意力机制的梯度计算(Backward pass)
- 针对不同 GPU 架构优化性能
- 支持对齐的内存访问模式
- 通过 `advance_to_block()` 进行 block 级别的调度管理
