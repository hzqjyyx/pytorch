这个Python脚本是一个**CUDA kernel代码生成器**，用于为memory-efficient attention算法生成大量kernel组合。

## 核心功能

### 1. 定义kernel参数空间
- **数据类型**: float32, float16, bfloat16
- **GPU架构**: SM50→70→75→80→100
- **内存对齐**: aligned/not-aligned
- **Block尺寸**: 如64x64, 64x128等不同的Q/K维度组合
- **功能开关**: dropout支持、bias支持

### 2. Forward Kernel生成 (FwdKernel类)

**参数组合逻辑**:
- 通过`itertools.product`生成所有可能的参数组合
- 过滤无效组合（如SM50不支持bfloat16，SM80+不需要unaligned kernel）
- 针对不同架构选择最优配置（如A100用64x128而非32x128）

**Kernel优先级排序** (`sort_index` in aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/generate_kernels.py:66):
```python
self.sort_index = (
    0 if self.aligned else 1,      # 优先aligned
    self.max_k,                      # 其次输出保持在寄存器
    self.k,
    1 if self.supports_dropout else 0,  # 无dropout更优
    1 if self.supports_bias else 0,
)
```

**生成内容**:
- Kernel函数名: 如`fmha_cutlassF_f16_aligned_64x64_rf_sm80`
- C++模板实例化: `AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, ...>`
- CUDA kernel实现: 带有架构检查的`__global__`函数

### 3. 代码输出结构

**`write_decl_impl`函数**生成两类文件:

**A. 头文件** (`cutlassF.h`):
- 所有kernel的函数声明
- 按数据类型和SM版本分组的dispatch函数
- 顶层`dispatch_cutlassF<DT>(callback, cc)`模板，运行时根据数据类型和compute capability选择kernel

**B. 实现文件** (多个`.cu`文件):
- 按`impl_group`分组（如`f16_aligned.cu`, `bf16_notaligned.cu`）
- 每个文件包含该组所有kernel的实际实现
- 使用`KERNEL_IMPL_TEMPLATE`生成带架构guard的kernel代码

### 4. Dispatch机制

生成的dispatch函数采用**回调模式**:
```cpp
template <typename DT, typename T>
void dispatch_cutlassF(T cb, int cc) {
    if (std::is_same_v<DT, cutlass::half_t> && 80 <= cc && cc < 100) {
        dispatch_cutlassF_f16_sm80(cb, cc);
    }
}
```
用户传入callback，系统遍历已排序的kernel列表，调用第一个支持当前输入的kernel。

### 5. 关键优化策略
- **Accumulator位置**: `max_k <= k`时输出保持在寄存器(RF)，否则用全局内存(GMEM)
- **架构特化**: SM80+的half类型启用MMA预加载(`preload_mmas`)
- **Shared memory**: SM80和SM70-half支持128的block_i（需更多shmem）

---

**ROCm相关**: 无（此文件纯CUDA）

**Backward相关**: 
- `BwdKernel`类: 类似结构，额外参数包括`block_i/j`, `preload_mmas`, `keys_queries_aligned_to_blocksizes`
- 特化kernel: 为Stable Diffusion (K=80)优化的128x64配置
- 生成`cutlassB.h`和对应`.cu`文件
