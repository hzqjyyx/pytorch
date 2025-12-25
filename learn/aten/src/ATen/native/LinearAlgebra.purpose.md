## LinearAlgebra.cpp 和 LinearAlgebra.h 主要功能

这两个文件实现了 PyTorch ATen 库的线性代数运算核心功能。

### 核心功能模块

**矩阵乘法系列**
- `mm` / `addmm` - 矩阵乘法及加法融合（支持 CPU BLAS、MKLDNN、ACL 优化）
- `bmm` / `baddbmm` - 批量矩阵乘法
- `matmul` - 通用矩阵乘法（自动处理维度广播，支持 1D-2D 各种组合）
- `mv` - 矩阵向量乘法
- `dot` / `vdot` / `inner` / `outer` - 向量运算

**矩阵分解与求逆**
- `linalg_det` / `_linalg_det` - 行列式计算（基于 LU 分解）
- `linalg_slogdet` - 符号对数行列式
- `linalg_pinv` - Moore-Penrose 伪逆（支持 Hermitian 优化，SVD/特征值分解）
- `linalg_inv` / `linalg_tensorinv` - 矩阵求逆与张量求逆
- `linalg_solve` / `linalg_tensorsolve` - 线性方程组求解

**矩阵幂与指数**
- `linalg_matrix_power` - 矩阵幂运算（二进制分解优化）
- `linalg_matrix_exp` - 矩阵指数（Taylor 展开，度数 1/2/4/8/12/18 自适应）
- `matrix_exp_backward` - 矩阵指数反向传播

**范数计算**
- `linalg_vector_norm` - 向量范数（支持 0/1/2/inf 范数）
- `linalg_matrix_norm` - 矩阵范数（Frobenius / 核范数 / 2范数基于 SVD）
- `linalg_norm` - 通用范数接口
- `linalg_cond` - 条件数计算

**高级运算**
- `linalg_multi_dot` - 多矩阵乘法链优化（动态规划最优括号化）
- `chain_matmul` - 已弃用的矩阵链乘法
- `kron` - Kronecker 积
- `addr` - 秩1更新 (β*self + α*vec1⊗vec2)
- `linalg_matrix_rank` - 矩阵秩（基于奇异值/特征值）

**量化矩阵运算**
- `_int_mm` - int8 矩阵乘法（VNNI 优化）
- `_weight_int4pack_mm` - int4 权重矩阵乘法
- `_weight_int8pack_mm` - int8 打包权重乘法
- `_dyn_quant_matmul_4bit` - 动态量化 4bit 矩阵乘法

### 性能优化策略

**BLAS 调度**
- CPU BLAS（gemm/gemm_batched_with_stride）
- MKLDNN（ARM Neoverse / AArch64 ACL）
- 内存布局优化（F-contiguous / transpose 检测）

**批处理优化**
- `should_fold` - 自动判断是否将高维张量折叠为 2D 矩阵
- 批量维度广播与扁平化
- 小矩阵并行化（OpenMP）vs 大矩阵 BLAS 调用

**数值稳定性**
- TF32 禁用 (NoTF32Guard)
- scale-and-square 技术（矩阵指数）
- 自适应算法选择（atol/rtol 容差）

### 简要提及内容

**ROCm 相关**
- 部分函数有 ROCm 特定实现分支

**Backward 相关**
- `matrix_exp_backward` 使用解析函数自动微分
- `linalg_matrix_exp_differential` 已在 FunctionsManual.cpp 实现
- 多数操作的反向传播在其他文件定义
