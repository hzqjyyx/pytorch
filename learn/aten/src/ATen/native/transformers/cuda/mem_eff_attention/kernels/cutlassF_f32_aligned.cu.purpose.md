这个文件定义了一系列针对不同CUDA计算能力(SM架构)的内存高效注意力机制(Memory-Efficient Attention)的CUDA kernel实例化。

## 主要功能

文件为float32类型的对齐数据实例化了多个前向注意力kernel，通过模板参数控制：

1. **架构特化** - 针对不同NVIDIA GPU架构生成专门优化的kernel：
   - SM50 (Maxwell): GTX 900系列
   - SM70 (Volta): V100
   - SM75 (Turing): RTX 20系列, T4
   - SM80 (Ampere): A100, RTX 30系列

2. **Tile尺寸配置** - 不同的工作块划分策略：
   - 64x64: 小块处理，适合较短序列
   - 32x128 / 64x128: 长条形块，适合不同序列长度

3. **内存策略** - 两种中间结果存储方式：
   - `rf` (register file): 使用寄存器存储，适合短序列(kMaxK=64/128)
   - `gmem` (global memory): 使用全局内存，支持超长序列(kMaxK=65536)

4. **运行时架构检查** - 每个kernel通过`__CUDA_ARCH__`宏确保：
   - 仅在目标架构范围内执行
   - 架构不匹配时打印错误信息并返回
   - 早期返回优化：空block通过`advance_to_block()`提前退出

## 代码模式

每个kernel遵循统一结构：
```cuda
__global__ void __launch_bounds__(threads, blocks_per_sm)
fmha_cutlassF_f32_aligned_{tile}_{memory}_{arch}(Params p) {
  #ifdef __CUDA_ARCH__
  #if __CUDA_ARCH__ >= min && __CUDA_ARCH__ < max
    if (!p.advance_to_block()) return;
    AttentionKernel<...>::attention_kernel(p);
  #endif
  #endif
  // 错误处理
}
```

`__launch_bounds__`通过`kNumThreads`和`kMinBlocksPerSm`向编译器提供占用率提示，优化寄存器分配。

## 技术特点

- **零拷贝实例化**: 所有kernel共享`AttentionKernel`模板实现，此文件仅做类型/参数实例化
- **编译时分支**: 架构检查在编译期完成，运行时无分支开销
- **自动生成**: 文件头注明由`generate_kernels.py`生成，便于维护多种配置组合

---

**其他内容**:
- 无ROCm相关代码
- 无Backward实现(纯前向kernel)
