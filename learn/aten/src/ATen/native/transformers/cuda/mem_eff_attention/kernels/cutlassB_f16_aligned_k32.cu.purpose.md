这个文件定义了多个 CUDA 内存高效注意力机制的反向传播内核函数，针对不同的 GPU 架构进行了优化：

• **文件性质**：自动生成的 CUDA 内核文件，包含 5 个全局函数，每个针对特定 SM（Streaming Multiprocessor）架构

• **内核函数**：
  - `fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm70` - SM70 架构优化版本
  - `fmha_cutlassB_f16_aligned_64x64_k32_sm50` - SM50-SM70 兼容版本
  - `fmha_cutlassB_f16_aligned_64x64_k32_sm70` - SM70 标准版本
  - `fmha_cutlassB_f16_aligned_64x64_k32_sm75` - SM75 架构版本
  - `fmha_cutlassB_f16_aligned_64x64_k32_sm80` - SM80+ 架构版本

• **核心功能**：实现注意力机制的反向传播计算，使用 CUTLASS（一个高性能线性代数 CUDA 模板库）

• **参数配置**：
  - 数据类型：`half_t`（FP16 半精度浮点）
  - 块大小：64×64
  - Key 维度对齐：k32
  - 部分内核启用序列对齐优化

• **运行时检查**：每个内核包含 CUDA 架构版本检测，确保只在目标 SM 架构上执行，否则输出错误信息

• **性能优化**：使用 `__launch_bounds__` 指定线程数和最小块数以提高 GPU 利用率
