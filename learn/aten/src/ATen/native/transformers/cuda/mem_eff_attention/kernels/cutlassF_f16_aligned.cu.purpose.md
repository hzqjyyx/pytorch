这个文件是自动生成的CUDA kernel实例化文件，为不同GPU架构提供基于Cutlass的memory-efficient attention前向传播kernel。

## 核心功能

文件定义了一系列针对不同NVIDIA GPU compute capability (sm_50到sm_100)的attention kernel变体，所有kernel使用：
- **数据类型**: `cutlass::half_t` (FP16)
- **内存对齐**: aligned访问模式
- **操作方向**: 前向传播 (使用`kernel_forward.h`)

## Kernel变体组织

每个kernel函数遵循相同的模式：

```cuda
__global__ void __launch_bounds__(...) 
fmha_cutlassF_f16_aligned_{config}_{arch}(Params p)
```

使用编译时架构检查确保kernel只在目标GPU上运行：
```cuda
#if __CUDA_ARCH__ >= X && __CUDA_ARCH__ < Y
  if (!p.advance_to_block()) return;
  AttentionKernel<...>::attention_kernel(p);
#endif
```

如果架构不匹配，输出错误信息。

## 配置维度

**三种tile配置**用于不同场景权衡：

1. **64x64_rf**: 64×64 query/key tile，使用register file (kMaxK=64)
   - sm50, sm70, sm75, sm80各一个实例

2. **32x128_rf / 64x128_rf**: 更宽的K维度 (kMaxK=128)  
   - 32×128: sm50, sm70, sm75
   - 64×128: sm80
   - 适合更长序列

3. **32x128_gmem**: 使用全局内存存储中间结果 (kMaxK=65536)
   - sm50, sm70, sm75, sm80各一个
   - 处理极长序列，突破寄存器/共享内存限制

## 架构覆盖

- **sm50-sm70** (Maxwell/Pascal): 早期GPU支持
- **sm70-sm75** (Volta): 首个Tensor Core架构  
- **sm75-sm80** (Turing): 改进的Tensor Core
- **sm80-sm100** (Ampere及更新): 当前主流架构

总计**12个kernel函数**，通过模板参数`AttentionKernel<dtype, arch, aligned, queries_per_block, keys_per_block, kMaxK, single_value_iteration, supports_dropout>`实例化不同配置。

---

**其他相关内容**:
- ROCm支持: 无(纯CUDA实现)
- Backward支持: 无(仅前向kernel)
