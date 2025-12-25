这个文件定义了三个 CUDA 全局函数，用于实现高效的 Attention 机制的**反向传播**计算。文件是自动生成的（基于 `generate_kernels.py`）。

**核心功能：**

- **三个kernel函数**，分别针对不同的 GPU 架构：
  - `fmha_cutlassB_f16_notaligned_64x64_k32_sm50`：SM50-SM70 架构
  - `fmha_cutlassB_f16_notaligned_64x64_k32_sm70`：SM70-SM75 架构
  - `fmha_cutlassB_f16_notaligned_64x64_k32_sm75`：SM75-SM80 架构

- **参数配置**：
  - 使用 `cutlass::half_t`（float16 数据类型）
  - 不对齐的访存（notaligned）
  - 块大小 64×64，head 维度 32
  - 多个 false 标志控制 dropout、dropout_p 等功能

- **执行逻辑**：
  - 检查 CUDA 架构版本是否匹配
  - 若匹配，调用 `AttentionBackwardKernel::attention_kernel()` 执行反向计算
  - 若不匹配，打印错误信息

- **优化特点**：
  - 使用 `__launch_bounds__` 指定线程和块数配置
  - 使用预处理指令 `#ifdef __CUDA_ARCH__` 实现编译时架构选择
