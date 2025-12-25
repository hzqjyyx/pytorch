- **文件类型**: 自动生成的 CUDA 核函数文件（efficient attention 的 backward pass）

- **核心功能**: 为 PyTorch 的内存高效注意力机制提供 CUDA backward 核函数实现

- **包含的核函数**:
  - `fmha_cutlassB_f32_notaligned_64x64_k128_sm50` - 针对 SM50-SM70 GPU 架构
  - `fmha_cutlassB_f32_notaligned_64x64_k128_sm70` - 针对 SM70-SM75 GPU 架构
  - `fmha_cutlassB_f32_notaligned_64x64_k128_sm75` - 针对 SM75-SM80 GPU 架构

- **参数配置**:
  - 数据类型: float32
  - block 尺寸: 64×64
  - key 维度: 128
  - 不对齐优化 (notaligned)

- **运行机制**:
  - 使用 CUTLASS 库实现的 AttentionBackwardKernel 模板
  - 每个核函数检查 GPU 架构的兼容性
  - 调用 `attention_kernel()` 执行具体的 backward 计算
  - 不兼容架构时输出 FATAL 错误提示

- **关键特性**: 通过架构特定化编译确保在不同 NVIDIA GPU 上的最优性能
