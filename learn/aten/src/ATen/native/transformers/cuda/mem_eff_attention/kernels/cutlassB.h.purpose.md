这个文件是 CUTLASS 库实现的前向 FMHA (Fused Multi-Head Attention) kernel 的实例化入口。

## 核心功能

**Kernel 实例化模板**
文件通过 `INSTANTIATE_ATTENTION_KERNEL_*` 宏批量实例化不同配置的 attention kernel：
- `scalar_t`: 数据类型（float, half, bfloat16）
- `ArchTag`: GPU 架构（SM50, SM70, SM75, SM80）
- `aligned`: 内存对齐要求（true/false）
- `Attention kernel type`: 不同的 kernel 实现变体

**主要 Kernel 类型**
- `AttentionKernel`: 基础 attention kernel
- `AttentionKernelBatched`: 批处理版本
- `AttentionKernelPaged`: 分页内存版本
- `AttentionKernelDecoder`: decoder 专用版本

**架构特化**
根据不同 CUDA 架构启用对应的 kernel：
- SM50: 基础支持
- SM70: Volta 架构优化
- SM75: Turing 架构优化  
- SM80: Ampere 架构优化（支持 TF32、异步加载等新特性）

**内存对齐策略**
同时实例化 aligned=true 和 aligned=false 版本，运行时根据输入张量的内存对齐情况选择最优 kernel。

**编译优化**
使用条件编译（`#ifndef __CUDA_ARCH__` 等）确保每个编译单元只实例化必要的 kernel 版本，避免编译时间和二进制体积爆炸。

---

**Backward 相关**: 文件中 `#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD` 保护的部分实例化了反向传播 kernel

**ROCm 相关**: `#ifdef __HIP_PLATFORM_AMD__` 部分为 AMD GPU 平台提供了对应的 kernel 实例化
