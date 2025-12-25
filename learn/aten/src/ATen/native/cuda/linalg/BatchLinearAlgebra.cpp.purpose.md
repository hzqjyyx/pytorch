这个文件是 PyTorch ATen 库中 CUDA 平台上批量线性代数运算的核心实现，主要提供各类矩阵分解和求解算法的 GPU 加速版本。

## 核心架构

**双后端支持机制**：
- 优先使用 **MAGMA 库**（Matrix Algebra on GPU and Multicore Architectures）处理批量运算
- 备选 **cuSOLVER/cuBLAS** 库处理单矩阵或小批量情况
- 通过 `at::globalContext().linalgPreferredBackend()` 运行时切换后端
- 批量大小/矩阵尺寸自适应路由（如 triangular_solve_kernel:1749 中：batch ≤ 8 且 size ≥ 64x64 用 cuBLAS forloop，否则根据 size ≤ 512 选择 cuBLAS batched 或 MAGMA batched）

## 实现的算法

### 1. **矩阵分解类**
- **Cholesky 分解**（cholesky_kernel:1320）：`A = L L^H`，支持上三角/下三角存储
- **LDL 分解**（ldl_factor_kernel:1102）：`A = L D L^H`，对称/Hermitian 矩阵分解
- **LU 分解**（lu_factor:1530）：带/不带主元两种模式，批量/单矩阵两种路径
- **QR 分解**（geqrf_kernel:1846）：用于最小二乘和正交化
- **SVD 分解**（svd_kernel:2210）：奇异值分解，支持 full/reduced 模式

### 2. **特征值问题**
- **对称特征分解**（linalg_eigh_kernel:1994）：矩阵尺寸 > 128 用 MAGMA GPU 计算，≤ 128 回退 CPU
- **一般特征分解**（linalg_eig_kernel:2079）：处理非对称矩阵

### 3. **线性方程组求解**
- **Cholesky 求解**（_cholesky_solve_helper_cuda:1240）：`A x = b`，A 为正定矩阵
- **LU 求解**（lu_solve_kernel:2480）：一般线性系统
- **三角求解**（triangular_solve_kernel:1749）：性能优化策略复杂
- **最小二乘**（lstsq_kernel:2731）：overdetermined/underdetermined 系统

### 4. **正交矩阵生成**
- **orgqr**（orgqr_kernel_impl:1771）：从 QR 分解生成 Q 矩阵
- **ormqr**（ormqr_kernel:1787）：Q 矩阵与其他矩阵相乘

## 性能优化策略

**批量处理限制**：
- MAGMA batched API 限制 batch_size ≤ 65535（见 triangular_solve_batched_magma:1718）
- 大批量自动分片为 mini-batches 处理

**内存管理**：
- `ALLOCATE_ARRAY` 宏（:1031）使用 pinned memory 优化 CPU-GPU 传输
- 工作空间查询一次性完成（如 linalg_eigh_magma:1913）避免重复分配

**数据布局**：
- `cloneBatchedColumnMajor` 转换为 Fortran 列主序（MAGMA/cuSOLVER 要求）
- stride 检查确保 GPU kernel 高效访问

**类型派发**：
- `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES` 模板支持 float/double/complex<float>/complex<double>
- MAGMA 函数通过模板特化映射到对应 C API（如 magmaLu<double> → magma_dgetrf_gpu）

## 错误处理

- **MAGMA info 代码**（checkMagmaInternalError:1011）：
  - info = 0：成功
  - info < 0：第 -i 个参数非法
  - info > 0：算法特定错误（如奇异矩阵）
- **异常传播**：立即检查 info 并 early return，避免浪费后续计算（见 :1950）

## 注册机制

通过 `REGISTER_CUDA_DISPATCH` 宏将 14 个 kernel 注册到 PyTorch dispatcher，包括：
- ldl_factor/solve, cholesky/cholesky_inverse, lu_factor/lu_solve
- triangular_solve, orgqr/ormqr, geqrf
- linalg_eigh/eig, svd, lstsq

---

**ROCm 相关**：
- 条件编译检查 `USE_ROCM`，部分功能禁用 cuSOLVER 路径
- HIP 平台替代 CUDA API

**Backward 相关**：
- 本文件仅涉及 forward 计算
- 自动微分梯度通过其他 autograd 层处理
