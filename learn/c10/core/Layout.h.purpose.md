- **枚举定义**：定义了 `Layout` 枚举类型，包含 9 种张量存储格式：Strided（标准密集）、Sparse（稀疏）、SparseCsr/Csc/Bsr/Bsc（不同的压缩稀疏格式）、Mkldnn（Intel MKL-DNN 格式）、Jagged（参差不齐格式）、NumOptions（选项总数）

- **常量别名**：为各种 Layout 提供 `k` 前缀的常量别名（如 `kStrided`、`kSparse` 等），便于代码中使用

- **后端到布局映射**：`layout_from_backend()` 函数根据 Backend 类型推断对应的 Layout，如 SparseCPU/CUDA 映射到 Sparse，MkldnnCPU 映射到 Mkldnn

- **流输出操作**：重载 `operator<<` 使 Layout 枚举能以字符串形式输出到流中，用于日志和调试

- **核心职责**：为 PyTorch ATen 库定义张量的存储布局属性，支持多种优化的数据组织方式
