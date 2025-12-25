## BatchLinearAlgebra.h/.cpp 核心功能

这两个文件实现了 PyTorch 中**批量线性代数操作**的 CPU 和 LAPACK/BLAS 后端支持。

### 头文件 (BatchLinearAlgebra.h)

**主要定义：**

1. **LAPACK 函数模板声明** (仅在 `AT_BUILD_WITH_LAPACK()` 时启用)：
   - Cholesky 分解：`lapackCholesky`, `lapackCholeskyInverse`
   - 特征值分解：`lapackEig`, `lapackSyevd` (对称/Hermite 矩阵)
   - QR 分解：`lapackGeqrf`, `lapackOrgqr`, `lapackOrmqr`
   - 最小二乘求解：`lapackGels`, `lapackGelsd`, `lapackGelsy`, `lapackGelss`
     - 提供统一接口 `lapackLstsq` 和 `LapackLstsqDriverType` 枚举来选择不同驱动
   - LU 分解：`lapackLu`, `lapackLuSolve`
   - LDL 分解：`lapackLdlHermitian`, `lapackLdlSymmetric`, `lapackLdlSolveHermitian`, `lapackLdlSolveSymmetric`
   - SVD 分解：`lapackSvd`

2. **BLAS 函数模板声明** (仅在 `AT_BUILD_WITH_BLAS()` 时启用)：
   - 三角求解：`blasTriangularSolve`

3. **Dispatch Stub 声明**（多后端调度机制）：
   - 声明了 30+ 个 dispatch stub，用于 CPU/CUDA/MPS 等后端的动态分发
   - 例如：`cholesky_stub`, `lu_factor_stub`, `svd_stub`, `linalg_eig_stub` 等

### 实现文件 (BatchLinearAlgebra.cpp)

**核心内容：**

1. **LAPACK/BLAS 外部函数声明**（126-476 行）：
   - 声明 Fortran LAPACK 函数（`zgetrf_`, `dgeqrf_`, `spotrf_` 等）
   - Windows ARM64 平台特殊处理（wrapper 函数）

2. **模板特化实现**（857-1538 行）：
   - 为 `float/double/complex<float>/complex<double>` 实现所有 LAPACK 函数
   - 将 C++ 类型映射到 Fortran LAPACK 调用

3. **Torch 公开 API 实现**（1562-4100 行）：
   - **矩阵求逆**：`linalg_inv`, `linalg_inv_ex`, `inverse`
   - **Cholesky 分解**：`linalg_cholesky`, `linalg_cholesky_ex`, `cholesky_solve`
   - **线性方程求解**：`linalg_solve`, `triangular_solve`
   - **LU 分解**：`linalg_lu_factor_ex`, `linalg_lu`, `lu_unpack`, `linalg_lu_solve`
   - **QR 分解**：`linalg_qr`, `linalg_householder_product`, `orgqr`, `ormqr`
   - **特征值分解**：`linalg_eigh`, `linalg_eigvalsh`, `linalg_eig`, `linalg_eigvals`
   - **SVD 分解**：`linalg_svd`, `linalg_svdvals`
   - **最小二乘**：`linalg_lstsq`
   - **LDL 分解**：`linalg_ldl_factor_ex`, `linalg_ldl_solve`
   - **向量运算**：`linalg_vecdot`, `linalg_vander`

4. **Meta 函数**（478-843 行）：
   - 为每个操作定义 `TORCH_META_FUNC`，负责输出张量的形状推断和内存布局设置
   - 处理批量维度广播、优先列主序（column-major）布局以兼容 BLAS/LAPACK

5. **错误检查机制**（1562-1638 行）：
   - `_linalg_check_errors`：解析 LAPACK 返回的 `info` 值，转换为有意义的错误消息
   - 区分不同错误类型：非法参数（info < 0）、奇异矩阵、非正定、不收敛等

6. **批量处理逻辑**：
   - 使用循环对每个批次矩阵调用单个 LAPACK 函数
   - 支持任意批量维度（batch dimensions）

---

### 结尾简单列出（ROCm/Backward 相关）：

- **ROCm 支持**：文件中未直接涉及 ROCm 特定代码，通过 Dispatch Stub 机制支持 HIP 后端
- **Backward（反向传播）**：文件实现前向操作，梯度计算在 `derivatives.yaml` 和自动微分系统中定义
