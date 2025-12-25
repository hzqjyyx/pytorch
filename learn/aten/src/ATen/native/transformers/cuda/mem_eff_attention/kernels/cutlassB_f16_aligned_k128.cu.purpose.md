这个文件定义了一系列用于**内存高效注意力机制反向传播**的 CUDA kernel 实例化，针对 f16（half 精度）数据类型且 head dimension 为 128 的场景。

## 主要功能

文件通过模板实例化创建了多个专门优化的 attention backward kernels，每个 kernel 针对不同的：
- **GPU 架构**：Sm50, Sm70, Sm75, Sm80（对应不同代数的 NVIDIA GPU）
- **Block 配置**：64x64 或 128x64/128x128 的 tile 大小
- **优化选项**：是否启用 sequence alignment

## Kernel 命名规范

以 `fmha_cutlassB_f16_aligned_128x64_k128_seqaligned_sm70` 为例：
- `fmha`: Fused Multi-Head Attention
- `cutlassB`: Backward pass，基于 CUTLASS 库
- `f16_aligned`: 使用对齐的 half 精度数据
- `128x64`: 输出 tile 大小（M=128, N=64）
- `k128`: head dimension = 128
- `seqaligned`: 启用序列对齐优化（可选）
- `sm70`: 目标 GPU 计算能力

## 架构分发机制

每个 kernel 使用编译时架构检查确保在正确的 GPU 上运行：
```cuda
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  // 执行 sm70 kernel
#endif
```

如果架构不匹配，打印错误信息而非执行。

## 模板参数说明

`AttentionBackwardKernel` 的关键模板参数：
1. 架构标签（如 `cutlass::arch::Sm70`）
2. 数据类型（`cutlass::half_t`）
3. `is_aligned = true`：输入内存对齐
4. 第4个参数：通常为 false（功能标志）
5. 第5个参数：是否启用 SM80+ 特性（如异步拷贝）
6. Block M, N, K 尺寸
7. `kApplySeqAlignment`：序列对齐标志（可选）

## ROCm 相关内容
- 无（纯 CUDA 实现）

## Backward 相关内容
- 整个文件专门用于 attention 反向传播
- 包含多种 block size 配置以优化不同 batch/sequence 长度
- 使用 `AttentionBackwardKernel` 模板，计算 dQ, dK, dV 梯度
- 通过 `kNumThreads` 和 `kMinBlocksPerSm` 优化 GPU 占用率
