这个文件定义了四个 CUDA 全局函数，用于高效注意力机制的反向传播计算。每个函数针对不同的 GPU 架构进行优化：

**SM50 版本** (`fmha_cutlassB_f32_aligned_64x64_k32_sm50`)
- 目标架构：SM 5.0-7.0（Kepler、Maxwell、Pascal GPU）
- 线程配置：64×64 块，key 维度 32

**SM70 版本** (`fmha_cutlassB_f32_aligned_64x64_k32_sm70`)
- 目标架构：SM 7.0-7.5（Volta、Turing GPU）
- 线程配置：64×64 块，key 维度 32

**SM75 版本** (`fmha_cutlassB_f32_aligned_64x64_k32_sm75`)
- 目标架构：SM 7.5-8.0（Turing、Ampere GPU）
- 线程配置：64×64 块，key 维度 32

**SM80 版本** (`fmha_cutlassB_f32_aligned_64x64_k32_sm80`)
- 目标架构：SM 8.0-12.0（Ampere、Ada Lovelace GPU）
- 线程配置：64×64 块，key 维度 32

**核心特点：**
- 基于 CUTLASS 库实现的模板化内核
- 所有版本使用 float32 精度、对齐内存访问
- 运行时架构检查：如果实际 GPU 架构不匹配，输出错误信息
- 自动生成代码（见文件头注释）
- 采用分块处理策略优化 GPU 内存访问模式
