# custom_mma_multistage.h 主要功能分析

这是一个基于 CUTLASS 库的**多阶段流水线式矩阵乘法（GEMM）**实现，专门用于 CUDA Tensor Cores 的 threadblock 级别计算。

## 核心架构

**CustomMmaMultistage 类**继承自 CustomMmaBase，实现了一个高性能的 GEMM 内核，关键特性：

- **软件流水线（Software Pipelining）**：使用多阶段（multi-stage）缓冲机制
- **异步内存拷贝**：利用 `cp.async` 指令重叠数据传输和计算
- **双缓冲/多缓冲**：至少需要 2 个 warp-level GEMM 操作来支持流水线结构

## 数据流动路径

```
Global Memory (A/B) 
    ↓ [cp.async with cache operations]
Shared Memory (multiple stages)
    ↓ [warp tile iterators]
Register Fragments
    ↓ [warp-level MMA]
Accumulator (FragmentC)
```

## 关键执行阶段

### 1. Prologue（_prologue 函数，lines 370-451）
- 预加载 `kNumStagesConcurrentLoad` 个阶段的数据到 shared memory
- 使用 `cp.async_zfill` 处理边界情况（out-of-bounds 自动填零）
- 每个阶段加载完后调用 `cp_async_fence()` 标记边界

### 2. Mainloop（operator() 函数，lines 594-741）
核心循环逻辑：
```
for each k-iteration:
  for each warp_mma_k (warp-level GEMM iterations):
    - Load next fragments from shared memory (double buffering)
    - Transform fragments (layout conversion)
    - Execute warp MMA instruction
    - Issue global→shared copies for next stage (interleaved)
    - Wait and sync at stage boundaries
    - Update circular buffer pointers
```

## 性能优化技术

### 1. 异步拷贝优化（lines 292-367, copy_tiles_and_advance）
- **分组访问**：按 `kAccessesPerGroupA/B` 分组，与计算交错执行
- **缓存控制**：通过 `CacheOpA/B` 参数控制 L1/L2 缓存行为
- **条件零填充**：根据 `zero_outside_bounds_` 或 `SharedMemoryClear` 选择 `cp_async` 或 `cp_async_zfill`

### 2. 计算与访存重叠
- **双缓冲 fragments**：`warp_loaded_frag_A/B[2]` 和 `warp_transformed_frag_A/B[2]`（lines 546-549）
- **提前加载**：在执行 MMA(k) 时加载 MMA(k+1) 所需数据
- **延迟同步**：`cp_async_wait<kNumStagesConcurrentLoad-1>()` 等待倒数第二个阶段，最大化并发

### 3. TF32x3 特殊处理（lines 574-589, 634-656）
- 使用临时累加器 `tmp_accum` 进行分阶段累加
- 针对 `OpMultiplyAddFastF32` 和 `OpMultiplyAddComplexFastF32` 的特化路径

## Shared Memory 管理

### 循环缓冲区逻辑（lines 702-725）
- **写指针**：`smem_write_stage_idx` 在 [0, kStages-1] 循环
- **读指针**：`smem_read_stage_idx` 在 [0, kStages-1] 循环
- **回绕处理**：达到最后阶段时，通过负偏移量 `add_tile_offset({0, -Base::kStages})` 返回起点

### 内存清零策略
- **kZfill**（lines 318-320）：使用 `cp_async_zfill` 在拷贝时自动填零
- **kClearLastStage**（lines 497-538）：显式清零最后一个阶段的 shared memory

## 边界条件处理

- **kSmemContainsEntireMat**（line 166）：判断 shared memory 是否容纳整个矩阵（`kMaxK <= Shape::kK * Stages`）
- **迭代次数计算**（line 286）：`iter = (problem_size_k + Shape::kK - 1) / Shape::kK` 向上取整
- **mask 控制**（lines 380-381, 562-563）：通过 `clear_mask(gemm_k_iterations == 0)` 禁用越界访问

## 架构要求

- **最低架构**：SM80（Ampere）用于支持 `cp.async` 指令（line 129）
- **Tensor Core 操作**：通过 `Policy::Operator` 封装的 warp-level MMA 指令

---

## ROCm 和 Backward 相关（简略）
- 文件中无 ROCm 特定代码（纯 CUDA/CUTLASS 实现）
- 无显式 backward 逻辑（这是前向 GEMM primitive，反向传播由上层调用不同配置的 GEMM）
- Memory-efficient attention 的梯度计算会复用此类，但需调整迭代器和累加顺序
