这个文件是自动生成的CUDA kernel声明文件，用于内存高效的注意力机制前向传播（memory-efficient attention forward pass）。主要内容包括：

## 核心架构

文件定义了一系列基于CUTLASS库的fused multi-head attention (FMHA) kernel声明，针对不同配置进行优化：

1. **数据类型支持**：
   - `bf16` (bfloat16)
   - `f16` (half precision)
   - `f32` (float)

2. **GPU架构支持**：
   - SM50 (Maxwell)
   - SM70 (Volta)
   - SM75 (Turing)
   - SM80 (Ampere)

3. **内存对齐变体**：
   - `aligned`: 内存对齐版本
   - `notaligned`: 非对齐版本

4. **Tile配置**：
   - `64x64`: 小块尺寸
   - `64x128` / `32x128`: 不同的块划分策略
   - 第三个维度参数（64/128/65536）控制内存使用策略（register file vs global memory）

## Kernel命名规范

每个kernel遵循命名模式：
```
fmha_cutlassF_{dtype}_{alignment}_{MxN}_{memory}_sm{arch}
```

例如 `fmha_cutlassF_f16_aligned_64x64_rf_sm80` 表示：
- f16数据类型
- 内存对齐
- 64x64 tile尺寸
- rf (register file) 内存策略
- SM80架构

## Dispatcher函数

文件提供分层调度机制：

1. **特定架构dispatcher**: `dispatch_cutlassF_{dtype}_sm{arch}(cb, cc)`
   - 为每个(数据类型, 架构)组合注册对应的kernel变体
   - 使用回调函数`cb`将kernel配置与实际kernel函数关联

2. **通用dispatcher**: `dispatch_cutlassF<DT>(cb, cc)`
   - 根据数据类型`DT`和compute capability `cc`自动选择合适的特定dispatcher
   - 使用`std::is_same_v`进行编译期类型检查
   - 使用compute capability范围判断选择架构版本

## AttentionKernel模板参数

每个kernel使用`AttentionKernel`模板实例化，参数含义：
```cpp
AttentionKernel<scalar_t, arch, is_aligned, kQueriesPerBlock, kKeysPerBlock, kMaxK, kIsAligned, kSingleValueIteration>
```

- 通过`kNumThreads`和`kMinBlocksPerSm`指定`__launch_bounds__`优化占用率

## 内存策略

根据第三个tile维度判断：
- **rf (register file)**: 值为64或128，数据存储在寄存器中，适合小规模
- **gmem (global memory)**: 值为65536，使用全局内存，适合大规模序列

---

**ROCm相关**：无（文件中未涉及AMD GPU支持）

**Backward相关**：无（此文件仅处理forward pass，对应的backward kernels在其他文件中定义）
