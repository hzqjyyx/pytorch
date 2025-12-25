这个文件实现了一个支持**动态量化(dequantization)的多阶段GEMM(矩阵乘法)**模板类 `DqMmaMultistage`，主要用于CUDA上的高效矩阵运算。

## 核心功能

**多阶段流水线架构**
- 使用多阶段(multi-stage)设计重叠数据传输和计算
- 通过 `cp.async` 指令异步地将数据从全局内存拷贝到共享内存
- 在数据传输的同时执行warp级别的矩阵乘法运算，隐藏内存延迟

**量化权重的动态反量化**
- 在计算过程中对量化的B矩阵进行实时反量化
- 通过 `warp_dequantizer_` 加载scale参数并应用到量化后的数据
- 支持在shared memory中存储scale数据以供warp访问

**数据流管理**
分为三个阶段：

1. **Prologue(前序阶段)**:
   - 加载scale数据到共享内存
   - 预填充前 `kStages-1` 个stage的A和B数据
   - 使用 `cp_async_fence()` 标记每个stage的边界

2. **Mainloop(主循环)**:
   - 从共享内存加载数据到warp fragment
   - 对B矩阵fragment执行 `TransformBAfterLDS` 转换
   - 调用 `dequantizer_.dequantize()` 应用scale
   - 执行 `run_warp_mma()` 完成矩阵乘累加
   - 同时触发下一个stage的全局到共享内存拷贝
   - 使用双缓冲(warp_frag[2])在加载和计算间切换

3. **循环缓冲区管理**:
   - 维护 `smem_write_stage_idx` 和 `smem_read_stage_idx`
   - 当索引达到 `kStages-1` 时回绕到0
   - 通过 `add_tile_offset` 负偏移实现循环

**架构要求**
- 最低要求SM80架构(Ampere)以支持 `cp.async` 指令
- 需要至少2个warp-level GEMM迭代来支持流水线
- 支持可选的tile interleaved layout用于特定数据排布

**内存优化选项**
- `SharedMemoryClearOption::kZfill`: 使用zfill模式处理越界访问
- `SharedMemoryClearOption::kClearLastStage`: 清零最后一个stage以确保GEMM范围外的累加器为0
- 通过 `CacheOp` 参数控制缓存策略

## 关键实现细节

- `copy_tiles_and_advance()`: 批量拷贝A和B的tile并推进迭代器
- 使用 `kAccessesPerGroupA/B` 将拷贝操作分组，与warp计算交错执行
- `cp_async_wait<kStages-2>()`: 确保流水线中始终有足够的数据可用
- 双fragment缓冲 + K维度分组加载实现计算与访存的精细重叠

---

**ROCm相关**: 无（此文件是NVIDIA CUTLASS扩展，仅用于CUDA）

**Backward相关**: 无（这是前向计算的GEMM kernel实现，不涉及梯度反向传播逻辑）
