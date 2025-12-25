这个文件提供了 PyTorch 线性代数操作的核心工具函数集合，主要包含：

## 内存布局和张量克隆

**批量矩阵的列优先处理**
- `batched_matrix_contiguous_strides`: 计算批量矩阵的连续步长，支持 Fortran（列优先）和 C（行优先）布局
- `cloneBatchedColumnMajor`: 将张量克隆为批量列优先格式，确保每个 (M, N) 矩阵都是列优先，且批次在内存中连续排列
- `copyBatchedColumnMajor`: 更快的替代方案，使用 `copy` 而非 `clone`，支持扩展行数和广播批次大小

**辅助函数**
- `expect_resolved_conj`: 处理共轭张量，必要时解析共轭
- `borrow_else_clone`: 条件性借用或克隆张量
- `same_stride_to`: 保留原始步长的设备/dtype 转换

## 批次处理

- `batchCount`: 计算批量矩阵的批次数量（除最后两维外所有维度的乘积）
- `matrixStride`: 计算单个矩阵的元素数量
- `batch_iterator_with_broadcasting`: 核心迭代器，处理带广播的批量操作，为 LAPACK/MAGMA 提供工作指针。实现了内存高效的广播策略——当 `a` 需要广播到 `b` 的形状时，维护一个缓冲区和访问标志，仅在必要时复制数据

## 输入验证

**形状检查**
- `checkIsMatrix`: 检查至少 2 维
- `squareCheckInputs`: 验证方阵（最后两维相等）
- `checkInputsSolver`: 验证线性求解器输入的兼容性（AX=B 或 XA=B）
- `checkAllSameDim`: 检查 TensorList 中所有张量维度一致

**类型检查**
- `checkFloatingOrComplex`: 验证浮点或复数类型，可选禁用低精度类型
- `checkNotComplexTolerance`: 确保容差张量不是复数
- `linearSolveCheckInputs`: 验证线性求解方法的形状和设备一致性
- `checkLinalgCompatibleDtype`: 检查输入输出类型的安全转换
- `checkUplo`: 验证上/下三角参数（'U'/'L'）

**设备检查**
- `checkSameDevice`: 确保张量在同一设备

## 广播操作

- `_linalg_broadcast_batch_dims`: 广播两个张量的批次维度，返回扩展后的尺寸或张量
- `broadcast_batch_size`: 计算两个张量的广播批次大小
- `BroadcastLinearIndices`: 类，用于高效访问广播张量的元素，维护线性索引映射

## 张量操作

**维度操作**
- `_move_to_end`: 将指定轴移动到张量末尾
- `create_dim_backshift_permutation`: 创建将两个维度移到末尾的排列
- `create_reverse_permutation`: 创建逆排列

**布局检查**
- `is_row_or_column_contiguous`: 检查行或列连续性
- `is_blas_compatible_column_major_order`: 检查是否与 BLAS 兼容的列优先布局（考虑 leading dimension 和批次步长）
- `is_blas_compatible_row_major_order`: 检查是否与 BLAS 兼容的行优先布局

## 特定操作工具

**QR 分解**
- `_parse_qr_mode`: 解析 QR 模式（"reduced"/"complete"/"r"）
- `_compute_geometry_for_Q`: 计算 Q 矩阵的大小、步长和列数

**SVD**
- `svd_uses_cusolver`: 判断是否使用 cuSOLVER
- `computeLRWorkDim`: 计算复数 SVD（cgesdd/zgesdd）的实数工作数组大小，针对不同平台（Apple Accelerate vs 标准 LAPACK）

**其他**
- `_get_epsilon`: 获取浮点类型的 epsilon 值
- `to_transpose_type`: 将连续性和共轭标志转换为转置类型枚举
- `linalg_solve_is_vector_rhs`: 判断线性求解的右侧是向量还是矩阵
- `get_linear_indices`: 计算广播访问的线性索引
- `checkUplo`: 验证 UPLO 参数（上/下三角）

---

**忽略内容（bullet-point）：**
- 无 ROCm 特定内容
- 无显式 Backward 相关代码（此文件为前向操作的工具层）
