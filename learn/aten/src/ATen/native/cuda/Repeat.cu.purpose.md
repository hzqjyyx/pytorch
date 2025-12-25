**CUDA Repeat Interleave 操作的实现**

该文件实现了 PyTorch 中 `repeat_interleave` 在 CUDA 上的计算逻辑。

**核心机制：**

`compute_cuda_kernel` 是主要的 GPU 内核函数：
- 接收一个 repeat 数组（表示每个元素的重复次数）和累积和数组
- 通过 warp 级别的并行化策略填充结果数组
- 每个 warp 处理一个元素的重复写入操作，线程在 warp 内协作填充多个位置

例如：repeat = [2, 3, 1]，则输出为 [0, 0, 1, 1, 1, 2]

**执行流程：**

1. `repeat_interleave_cuda` 函数接收 repeat 张量和可选的输出大小
2. 通过类型分发宏（`AT_DISPATCH_INDEX_TYPES`）处理不同的索引类型
3. 调用 `repeat_interleave_common` 模板函数（来自 Repeat.h），最后将 `compute_cuda` 作为计算策略传入
4. `compute_cuda` 配置 GPU 网格（grid）和块（block）参数，启动内核

**关键设计：**

- **Warp 级并行**：每个 warp 处理一个数组元素，充分利用 GPU 的向量化能力
- **动态 grid 大小**：基于数组大小和 warp 数量动态计算，上限 2048 个 block
- **索引类型通用**：通过模板支持 int32/int64 等不同索引类型

**概括：**

- GPU 上高效计算 repeat_interleave 操作（展开数组）
- Warp 级并行策略优化吞吐量
- 通用的模板设计支持多种数据类型
- 集成到 ATen 的算子分发系统
