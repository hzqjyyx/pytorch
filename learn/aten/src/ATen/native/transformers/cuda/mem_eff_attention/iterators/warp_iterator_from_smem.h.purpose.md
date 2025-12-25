## WarpIteratorFromSmem 核心功能

这是一个 CUDA/Cutlass 库中用于高效加载共享内存（shared memory）中张量数据的迭代器类。

**主要职责：**

- **从共享内存加载数据到寄存器**：使用 `ldmatrix` 指令（比手动加载快）将 32x32 的矩阵块分片加载到线程寄存器中

- **支持转置操作**：可以在读取时转置操作数（如读取 `A.transpose()` 当共享内存中存储的是 `A`）

- **为 A100 TensorCore 提供支持**：针对 16x8x8 和 16x8x16 的指令形状优化

- **线程级别坐标管理**：根据 lane_id（warp 内的线程 ID）计算每个线程应该加载的共享内存地址偏移

- **迭代控制**：通过 `operator++()` 和 `advance()` 方法遍历矩阵块的不同部分

**关键设计点：**

- 固定为 32x32 矩阵块（`Shape`）
- 仅支持 16-bit 元素（half/float16）
- 每个线程加载 4 个 32-bit 的值（`ldmatrix.x4`）
- 支持 kOperand::kA 和 kOperand::kB 两种操作数类型
- 行主序（RowMajor）内存布局
