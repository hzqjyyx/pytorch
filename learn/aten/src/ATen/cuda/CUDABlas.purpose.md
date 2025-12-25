## CUDABlas.h 和 CUDABlas.cpp 主要功能

提供 CUDA BLAS (Basic Linear Algebra Subprograms) 操作的 C++ 模板接口，封装 cuBLAS 库调用。

### 核心组件

**PointerModeGuard (CUDABlas.h:23-38)**
- RAII 守卫类，自动管理 cuBLAS 指针模式
- 构造时设置新模式，析构时恢复原模式
- 防止指针模式设置泄漏

### BLAS Level 3 (矩阵-矩阵运算)

**gemm (General Matrix Multiply)**
- 模板函数，支持类型：`double`, `float`, `c10::complex<double/float>`, `Half`, `BFloat16`
- 计算：`C = alpha * op(A) * op(B) + beta * C`
- 两个版本：`gemm` 和 `gemm_internal` (内部使用)
- 参数通过宏 `CUDABLAS_GEMM_ARGTYPES` 定义，包含转置标志、矩阵维度、缩放因子和指针

**bgemm (Batched GEMM)**
- 批量矩阵乘法，额外支持 stride 参数
- 处理多个矩阵对的乘法操作
- 同样提供 `bgemm` 和 `bgemm_internal` 版本

**gemm_and_bias**
- GEMM + bias 加法 + 激活函数融合操作
- 支持激活：None, ReLU, GELU (CUDA 11.4+)
- 通过 `GEMMAndBiasActivationEpilogue` 枚举控制

**特殊 GEMM 变体**
- `int8_gemm`: INT8 矩阵乘法，输出 INT32
- `scaled_gemm`: 支持量化缩放因子的矩阵乘法，处理混合精度和行/列缩放

**trsm/trsmBatched (Triangular Solve)**
- 求解三角矩阵方程：`op(A) * X = alpha * B`
- 支持单次和批量操作

### BLAS Level 2 (矩阵-向量运算)

**gemv (General Matrix-Vector Multiply)**
- 计算：`y = alpha * op(A) * x + beta * y`
- 支持类型同 gemm

### BLAS Level 1 (向量-向量运算)

**dot/vdot**
- `dot`: 标准点积，支持实数和复数类型
- `vdot`: 共轭点积，仅复数类型
- 不支持 BFloat16 (仅 ROCm)

### 线性代数求解器 (Batched)

**getrfBatched (LU Factorization)**
- 批量 LU 分解，用于求解线性系统

**getrsBatched (Linear System Solve)**
- 使用 LU 分解结果求解 `A * X = B`

**geqrfBatched (QR Factorization)**
- 批量 QR 分解

**gelsBatched (Least Squares)**
- 批量最小二乘问题求解

### 实现模式

1. **模板特化**：基础模板使用 `static_assert(false)` 触发编译错误，仅特化版本可用
2. **宏定义参数列表**：通过 `CUDABLAS_*_ARGTYPES` 统一参数签名
3. **类型映射**：使用 `at::opmath_type<Dtype>` 处理低精度类型的累加精度 (如 Half 用 float 累加)

---

**ROCm 相关**：文件中包含 ROCm 平台的 hipBLAS 实现分支

**Backward 相关**：线性代数求解器用于反向传播中的梯度计算
