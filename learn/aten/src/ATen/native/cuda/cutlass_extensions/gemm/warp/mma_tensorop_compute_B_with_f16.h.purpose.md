这是一个 CUTLASS 库的 CUDA 核心文件，实现了针对 Tensor Cores 的 warp 级矩阵乘法累加操作。

**主要功能：**

- **MmaTensorOpComputeBWithF16 类**：专门为半精度浮点数（FP16/BF16）矩阵乘法优化的计算内核
- **矩阵乘法操作**：实现 D = A × B + C 的 warp 级计算，其中 A、B、C 是矩阵分块
- **B 矩阵扩展**：支持将 B 矩阵的 K 维度扩展处理（通过 `kExpansionFactor`），用于特殊的共享内存布局
- **架构适配**：支持 SM75+（Turing）和 SM80+（Ampere）GPU，使用不同的遍历策略优化
  - SM75 使用列优先遍历（最大化 Rb 寄存器重用）
  - SM80+ 使用行优先遍历（最大化 Ra 寄存器重用）
- **灵活的累加器布局**：支持行优先和列优先两种累加器存储方式
- **Tile 迭代器**：通过 IteratorA/IteratorB/IteratorC 从共享内存中高效加载矩阵分块
- **类型支持**：仅支持 FP16 或 BF16（Ampere+），强制通过 static_assert 验证
- **K 分割**：支持沿 K 维度的分区并行（PartitionsK 参数）
