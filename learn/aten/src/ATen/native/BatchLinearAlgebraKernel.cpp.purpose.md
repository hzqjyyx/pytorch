# BatchLinearAlgebraKernel.cpp 主要功能

这个文件实现了 PyTorch 中批量线性代数运算的 CPU 内核函数，是 ATen 库的一部分。所有函数都通过调用 LAPACK/BLAS 库来完成实际计算。

## 核心功能模块

### 1. Cholesky 分解 (`apply_cholesky`, `cholesky_kernel`)
- 计算对称/Hermitian 正定矩阵的 Cholesky 分解
- 支持上三角或下三角模式
- 调用 LAPACK 的 POTRF 例程
- 批量处理：遍历 batch 维度，对每个矩阵单独分解

### 2. Cholesky 逆矩阵 (`apply_cholesky_inverse`, `cholesky_inverse_kernel_impl`)
- 基于 Cholesky 分解计算正定矩阵的逆
- 调用 LAPACK 的 POTRI 例程
- 使用 `apply_reflect_conj_tri_single` 填充三角矩阵的另一半（LAPACK 只写一半）

### 3. 特征值分解

**非对称矩阵** (`apply_linalg_eig`, `linalg_eig_kernel`):
- 计算一般矩阵的特征值和特征向量
- 调用 LAPACK 的 GEEV 例程
- 只计算右特征向量
- 工作空间优化：先查询最优 lwork，再分配

**对称/Hermitian 矩阵** (`apply_lapack_eigh`, `linalg_eigh_kernel`):
- 调用 LAPACK 的 SYEVD 例程（分治算法）
- 效率更高，适用于对称矩阵
- 特征值保证为实数

### 4. QR 分解相关

**QR 分解** (`apply_geqrf`, `geqrf_kernel`):
- 调用 LAPACK 的 GEQRF 例程
- 输出紧凑格式：R 存储在上三角，Q 的 Householder 反射向量存储在下三角和 tau 中

**构造正交矩阵 Q** (`apply_orgqr`, `orgqr_kernel_impl`):
- 从 Householder 反射向量重建正交矩阵 Q
- 调用 LAPACK 的 ORGQR/UNGQR 例程

**Q 矩阵乘法** (`apply_ormqr`, `ormqr_kernel`):
- 用 Q 或 Q^T/Q^H 乘以其他矩阵，无需显式构造 Q
- 调用 LAPACK 的 ORMQR 例程
- 支持左乘/右乘，转置/不转置

### 5. 最小二乘法求解 (`apply_lstsq`, `lstsq_kernel`)
- 求解最小二乘问题 min ||B - AX||
- 支持 4 种 LAPACK 驱动程序：
  - `GELS`: 基于 QR 或 LQ 分解（不计算秩）
  - `GELSY`: QR 分解 + 列主元
  - `GELSD`: 分治 SVD（推荐，快速）
  - `GELSS`: 传统 SVD
- 支持张量广播（A 和 B 可以有不同的 batch shape）

### 6. 三角线性方程组求解 (`apply_triangular_solve`, `triangular_solve_kernel`)
- 求解 op(A)X = B，其中 A 是三角矩阵
- 调用 BLAS 的 TRSM 例程
- 支持上/下三角、转置、单位对角线等选项

### 7. LDL 分解

**分解** (`apply_ldl_factor`, `ldl_factor_kernel`):
- 对称/Hermitian 矩阵的 Bunch-Kaufman 分解：A = LDL^T 或 A = LDL^H
- 调用 LAPACK 的 SYTRF/HETRF 例程
- 支持选择对称或 Hermitian 模式

**求解** (`apply_ldl_solve`, `ldl_solve_kernel`):
- 使用 LDL 分解求解线性方程组
- 调用 LAPACK 的 SYTRS/HETRS 例程

### 8. LU 分解

**分解** (`apply_lu_factor`, `lu_factor_kernel`):
- 计算 LU 分解并返回主元信息
- 调用 LAPACK 的 GETRF 例程
- 使用并行化优化：根据矩阵大小动态调整 grain size
- 启发式公式：`chunk_size = min(1.0, 3200.0 / rank³)`

**求解** (`apply_lu_solve`, `lu_solve_kernel`):
- 使用 LU 分解求解 AX = B
- 调用 LAPACK 的 GETRS 例程
- 支持广播：LU 和 pivots 可以广播到 B
- 包含 pivots 有效性检查（范围在 1 到 n 之间）

### 9. SVD 奇异值分解 (`apply_svd`, `svd_kernel`)
- 调用 LAPACK 的 GESVD 例程
- 支持完整/缩减模式（full_matrices 参数）
- 可选只计算奇异值（不计算 U, Vh）
- 复数矩阵需要额外的 rwork 工作空间

### 10. 主元解包 (`unpack_pivots_cpu_kernel`)
- 将 LU 分解的主元索引转换为置换矩阵
- 通过交换操作重建完整的行置换
- 包含 pivots 范围检查

## 设计模式

每个功能都遵循三层结构：

1. **核心实现函数** (`apply_xxx<scalar_t>`): 
   - 模板函数，处理具体数值类型
   - 调用 LAPACK/BLAS 接口
   - 批量循环处理

2. **类型分发函数** (`xxx_kernel`):
   - 使用 `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES` 宏
   - 根据张量的 scalar_type 分发到对应的模板实例

3. **注册宏** (`REGISTER_ARCH_DISPATCH`):
   - 为不同 CPU 架构（AVX2, AVX512, VSX, SVE256 等）注册对应的内核
   - 支持架构特定的优化

## 性能优化技术

- **工作空间查询**: 先调用 LAPACK 查询最优 workspace 大小（lwork = -1）
- **批量外层循环**: 只查询一次工作空间，所有 batch 共享
- **并行化**: LU 分解使用 `at::parallel_for` 并行处理 batch 维度
- **小矩阵优化**: 如 `apply_reflect_conj_tri_single` 中，n < 256 时避免 OpenMP 开销
- **避免不必要分配**: 如 `orgqr` 中检查 `numel() == 0` 提前返回

## 错误处理

- LAPACK 不可用时返回友好错误信息（`AT_BUILD_WITH_LAPACK` 宏）
- 使用 `TORCH_INTERNAL_ASSERT_DEBUG_ONLY` 检查 LAPACK 返回的 info 参数
- 对称特征值分解中，遇到错误提前返回避免浪费计算

---

**ROCm 相关内容**: 无（此文件仅针对 CPU）

**Backward 相关内容**: 无（此文件只包含前向计算内核）
