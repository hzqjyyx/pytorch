- **文件用途**: 提供 CUDA 张量索引信息的工具函数，用于在 GPU 核函数中高效访问张量数据

- **核心功能**:
  - `maybeOverlappingIndices()`: 检测张量是否存在重叠索引（声明）
  - `getTensorInfo()`: 将 PyTorch 张量转换为 GPU 可用的 `TensorInfo` 结构体，包含数据指针、维度、尺寸和步长信息

- **关键特性**:
  - 使用模板支持不同的数据类型和索引类型（32位/64位）
  - 通过 `canUse32BitIndexMath()` 优化 32 位索引的性能
  - 利用 `if constexpr` 在编译时区分常量和可变数据指针
  - 最多支持 `MAX_TENSORINFO_DIMS` 个维度

- **应用场景**: CUDA 核函数需要直接访问张量数据时，通过此函数快速获取所需的索引和指针信息
