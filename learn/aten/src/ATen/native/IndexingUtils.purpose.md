## IndexingUtils.cpp 和 IndexingUtils.h 功能分析

### 主要功能概述

这两个文件提供了 PyTorch 张量索引操作的工具函数库，用于处理复杂的索引操作、类型检查和张量变换。

### 详细功能说明

**IndexingUtils.cpp** 包含：
- `canUse32BitIndexMath()` - 判断张量是否可以用 32 位整数进行索引计算。通过计算张量的最大线性索引偏移量，来确定是否超过 32 位整数的表示范围。这在 GPU 计算中很重要，因为 32 位索引运算比 64 位更高效。

**IndexingUtils.h** 包含的工具函数：

1. **索引验证**
   - `invalid_mask()` - 抛出异常，报告掩码形状与张量形状不匹配的错误
   - `checkIndexTensorTypes()` - 验证用于索引的张量的数据类型（要求为 long、byte、bool 或 int）

2. **索引张量转换**
   - `expandTensors()` - 将 ByteTensor 或 BoolTensor 掩码转换为等价的长整型张量索引（通过 nonzero 操作）
   - `toListOfOptionalTensors()` - 将 ArrayRef 或 IValue 列表转换为可选张量列表

3. **张量重排**
   - `hasContiguousSubspace()` - 检查非空索引张量是否形成连续的子空间
   - `transposeToFront()` - 将张量和索引一起转置，使所有非空索引位于张量的前 k 个维度
   - `transposeToFrontAndInvPerm()` - 同上，但额外返回逆排列向量以恢复原始维度顺序

4. **高级索引结构**
   - `AdvancedIndex` - 存储处理后的源张量、索引、大小、步长和维度信息的结构体

### 核心要点

- 优化索引计算的数值精度（32 位 vs 64 位）
- 掩码转换成标准索引格式
- 索引张量类型检查和验证
- 张量维度重排以支持高效的索引操作
- 支持高级（advanced）索引的基础设施
