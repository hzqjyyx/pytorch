# DqMmaPipelined 核心功能

这是一个实现**双缓冲流水线化的量化矩阵乘法（Dequantized GEMM）**的 CUDA 模板类，用于在 CUDA 核心上高效执行混合精度矩阵运算。

## 关键设计特点

**双缓冲流水线架构**
- 使用 2 阶段（`kStages == 2`）的双缓冲机制
- 通过 `smem_write_stage_idx` 在两个共享内存缓冲区之间切换
- 重叠数据加载和计算：当前迭代计算时，预取下一迭代的数据

**量化权重处理流程**
- B 矩阵（权重）以量化格式存储在全局内存中
- 加载后立即应用 `TransformBAfterLDG` 转换（LDG = LoaD Global）
- 存入共享内存后再应用 `TransformBAfterLDS` 转换（LDS = LoaD Shared）
- 使用 `warp_dequantizer_` 在 warp 级别对量化数据进行反量化
- 支持缩放因子（scales）用于恢复原始精度

**三级存储层次**
1. **全局内存** → `IteratorA/B/Scale` 迭代器加载 tile 到寄存器片段
2. **共享内存** → `SmemIteratorA/B/Scale` 将片段写入共享内存，warp 级迭代器读取
3. **寄存器/计算** → Warp 级 MMA（Matrix Multiply-Accumulate）单元执行实际计算

## 主要执行流程

**Prologue（序幕阶段）**
```
加载第一个 tile：A, B, scales → 转换 → 存入共享内存
同步线程（__syncthreads）
加载第一对 warp 级片段到寄存器
```

**Mainloop（主循环）**
```
for 每个 K 维度迭代:
    for 每个 warp 内的子迭代:
        - 执行 warp_mma 计算当前片段
        - 预取下一个 warp 片段到寄存器（A 和 B 交替使用 2 个缓冲区）
        - 在最后一个子迭代时：
          * 将下一个 tile 写入共享内存
          * 同步线程
          * 循环缓冲区指针回绕
        - 对 B 片段应用反量化
```

## 性能优化技术

**隐藏延迟**
- 在 warp MMA 计算期间发起下一批数据的加载
- 全局内存加载、共享内存加载、计算三者流水线重叠

**内存访问优化**
- Tile 化访问模式，提高缓存命中率
- 使用共享内存作为暂存，减少全局内存访问延迟
- `clear_mask` 机制避免越界访问

**warp 级调度**
- 计算 warp 在 threadblock 中的 3D 坐标（M, N, K 维度）
- 为每个 warp 分配不同的 tile 偏移，实现并行
- A 矩阵按 warp_mma_k 迭代递增，B 矩阵每 `kNumKIterationsPerWarpBLoad` 次才加载新片段

## 模板参数语义

- `Shape_`: Threadblock 级别的 GEMM 形状（M×N×K）
- `IteratorA/B/Scale`: 全局内存迭代器，负责加载数据
- `SmemIteratorA/B/Scale`: 共享内存迭代器，负责存储/读取
- `Policy_`: 调优策略，包含 warp 级算子和分区配置
- `TransformBAfterLDG/LDS`: B 矩阵的两阶段转换，支持精度转换（如 BF16→FP16）

## 典型应用场景

用于 INT4/INT8 量化模型推理，特别是：
- 大语言模型（LLM）权重量化推理
- 需要在计算时反量化的混合精度场景
- FasterTransformer 等推理框架的底层算子

---

**忽略内容（ROCm/Backward）：**
- 此文件无 ROCm（HIP）相关代码
- 仅实现前向计算，无梯度反向传播逻辑
- 未涉及 autograd 或反向算子注册
