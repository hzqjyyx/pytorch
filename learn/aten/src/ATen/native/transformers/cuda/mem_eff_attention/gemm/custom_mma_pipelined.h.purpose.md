## 主要功能

这是一个实现了**双缓冲流水线(double-buffered pipelined)矩阵乘法**的 CUDA kernel 模板，专门用于在 GPU 的 CUDA cores 和 SIMT 架构上高效执行 threadblock 级别的 GEMM (General Matrix Multiply) 运算。

## 核心设计

**双缓冲机制** (line 96, 143-145)
- 强制要求 `kStages = 2`，实现双缓冲流水线
- 一个 buffer 用于计算，另一个同时进行数据加载
- 通过 `smem_write_stage_idx` (line 307, 360) 在两个 buffer 之间切换

**内存层次**
- Global memory → Shared memory: 通过 `IteratorA/B` 加载，`SmemIteratorA/B` 写入
- Shared memory → Registers: 通过 `warp_tile_iterator_A/B` 读取为 warp 级别的 fragment
- 数据转换: `TransformA/B` 在写入 shared memory 时进行类型转换 (line 283-284, 335-337)

## 执行流程

**Prologue** (line 263-289)
1. 初始化累加器 `accum = src_accum`
2. 预加载第一个 tile 的 A 和 B 数据到 shared memory
3. 同步线程 `__syncthreads()`
4. 预加载 warp 级别的第一个 fragment

**Mainloop** (line 322-392)
循环迭代 K 维度，每次迭代：
1. 执行 `kWarpGemmIterations` 次 warp 级别的矩阵乘加
2. 在最后一次 warp MMA 时 (line 333):
   - 将预加载的数据写入 shared memory
   - 同步线程
   - 切换 circular buffer 的读写指针
3. 在第一次 warp MMA 时 (line 374):
   - 从 global memory 预加载下一个 tile
   - 使用 `clear_mask` 避免越界读取
4. 双缓冲 warp fragments (`warp_frag_A/B[2]`) 重叠加载和计算

**延迟隐藏策略**
- Global memory 加载在 warp MMA 计算期间进行 (line 374-379)
- Shared memory 加载在前一个 warp MMA 完成后立即发起 (line 368-369)
- 计算、shared memory 访问、global memory 访问三者流水线并行

## 关键特性

**Warp 映射** (line 173-189)
- 将 warp_id 映射到 3D 坐标 (M, N, K)
- 为每个 warp 设置在 threadblock tile 内的起始偏移

**边界处理** (line 310-311, 382-383)
- 使用 `clear_mask` 在最后几次迭代时禁用无效的内存访问
- 提前两次迭代判断以适应流水线延迟

**模板灵活性**
- 支持不同的 Shape、Iterator、Data type、Layout 组合
- 通过 Policy 参数控制调优细节
- 支持 Complex transform (line 137-140)

---

**其他相关内容 (bullet points)**
- ROCm: 无相关内容
- Backward: 无相关内容，这是一个通用的前向 GEMM kernel 实现
- 继承自 `CustomMmaBase<Shape_, Policy_, 2>` 基类 (line 96, 99)
- `set_prologue_done()` 和 `set_zero_outside_bounds()` 在 pipelined 版本中未实现/不需要 (line 209-217)
