# EpiloguePipelined 核心功能分析

这是一个优化的 GEMM epilogue 实现，用于将矩阵乘法的累加结果写回全局内存，针对 Transformer 的 memory-efficient attention 场景。

## 主要优化点

基于 CUTLASS 的标准 epilogue 修改，增加三个关键优化：

1. **双缓冲流水线（Pipelining）**：同时加载 2 个 source fragments，在处理当前迭代时预取下一个
2. **支持不同数据类型读取**：source 和 output 可以是不同的 dtype
3. **行 ID 传递**：将行 ID 传递给 OutputOp（用于 normalization 等操作）

## 核心数据流

### 路径 1：不需要 source 数据 (`compute_source_not_needed_`)

```
Accumulator Tile (寄存器)
    ↓ [acc2smem_source_not_needed]
Shared Memory
    ↓ [SharedLoadIterator]
Aligned Accumulator Fragment
    ↓ [apply_output_operator_source_not_needed_]
Output Fragment
    ↓ [OutputTileIterator::store]
Global Memory
```

**流程**：
- 将累加器数据写入共享内存（按 `kFragmentsPerIteration` 分批）
- 从共享内存对齐加载
- 如果 `kPartitionsK > 1`，对多个 K 分区执行 reduction
- 应用 output operator（类型转换/激活函数等）
- 存储到全局内存

### 路径 2：需要 source 数据 (`compute_source_needed_`)

```
Source (全局内存)           Accumulator (寄存器)
    ↓ [预加载 iter 0]              |
    ↓ [load iter 1] ←──┐          ↓ [acc2smem]
    ↓                   │      Shared Memory
Source Fragment[0,1]   │          ↓
(双缓冲交替)            │   Aligned Fragment
    ↓                   │          ↓
    └──────→ [OutputOp] ←──────────┘
                ↓
         Output Fragment
                ↓
          Global Memory
```

**流水线机制**：
- 迭代 0 前：预加载 `source_fragment[0]`
- 迭代 i：
  - 加载 `source_fragment[(i+1)%2]`（下一个）
  - 同时处理当前 `source_fragment[i%2]`
  - 隐藏内存延迟

## 关键模板参数

- `Shape_`：threadblock tile 形状
- `WarpMmaOperator_`：warp 级 MMA 算子
- `PartitionsK`：K 维度分区数（用于并行 reduction）
- `OutputTileIterator_` / `OutputTileSourceIterator_`：输出/输入迭代器
- `OutputOp_`：输出算子（fusion 操作）
- `FragmentsPerPartition`：粗粒度控制参数

## 行偏移计算（`getRowOffset`）

通过嵌套循环遍历 ThreadMap 的层次结构：
```
cluster → group → row → column
```
根据 fragment 索引 `i` 找到对应的行偏移，用于：
- 传递给 OutputOp 做 row-wise 操作
- 支持类似 softmax normalization 的场景

## OutputOp 适配器（`ApplyEpilogueOp`）

提供两个重载：
- `apply(op, row_id, accum, source)`：需要 source 时
- `apply(op, row_id, accum)`：不需要 source 时

根据 OutputOp 的接口自动选择正确版本。

## 内存访问模式

- **共享内存分块**：`kSmemTiles` 个 tile，偏移量 `kSmemPointerOffset`
- **访问粒度**：`kElementsPerAccess` 个元素向量化访问
- **同步点**：`__syncthreads()` 在读写共享内存间同步

## 性能考虑

- `IterationsUnroll` 参数：OutputOp 较大时设为 0 减少二进制大小
- 静态断言检查：确保迭代器间的元素数量/迭代次数匹配
- 约束：`kPartitionsK == 1 || kFragmentsPerIteration == 1`（其一必须为 1）

---

**其他次要内容**：
- 文件基于 NVIDIA CUTLASS 库的 BSD-3-Clause 协议
- 支持 CUDA RTC 编译环境（`__CUDACC_RTC__`）
- 继承自 `EpilogueBase` 基类，复用共享存储和基础迭代器逻辑
- ROCm 相关：无特定代码
- Backward 相关：无特定代码（此 epilogue 主要用于 forward pass）
