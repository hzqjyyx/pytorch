# CUDA 二次 Dispatch 机制

以 matmul 为例，详细分析 dispatch 到 CUDA 后的再次分发机制。

---

## 1. 概念

"二次 dispatch" 指的是：第一次 dispatch（通过 native_functions.yaml）到达某个后端（如 CUDA）后，该后端内部再次根据条件选择不同的实现。

这种分发可以是：
- **运行时决策**：基于输入维度、数据类型、内存布局等
- **编译期决策**：基于模板特化

---

## 2. native_functions.yaml 定义

### 2.1 matmul 定义

**位置**: `native_functions.yaml:3775-3789`

```yaml
- func: matmul(Tensor self, Tensor other) -> Tensor
  variants: function, method
  dispatch:
    CompositeImplicitAutograd: matmul       # 组合实现
    NestedTensorCPU, NestedTensorCUDA: matmul_nested

- func: matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CompositeImplicitAutograd: matmul_out
    NestedTensorCPU, NestedTensorCUDA: matmul_out_nested
```

### 2.2 mm 定义

**位置**: `native_functions.yaml:4133-4150`

```yaml
- func: mm(Tensor self, Tensor mat2) -> Tensor
  structured_delegate: mm.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: _sparse_mm
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: _sparse_csr_mm

- func: mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CPU: mm_out_cpu
    CUDA: mm_out_cuda
    MPS: mm_out_mps
    XPU: mm_out_xpu
```

### 2.3 相关操作

**位置**: `native_functions.yaml:635-687`

```yaml
# 矩阵向量乘
- func: addmv.out(...)
  dispatch:
    CPU, CUDA: addmv_out

# 批量矩阵乘
- func: bmm.out(...)
  dispatch:
    CPU, CUDA: bmm_out_cuda

# 带偏置的批量矩阵乘
- func: baddbmm.out(...)
  dispatch:
    CPU, CUDA: baddbmm_out_cuda
```

---

## 3. 第一层分发：维度分析

### 3.1 Composite matmul 实现

**位置**: `aten/src/ATen/native/LinearAlgebra.cpp:2182-2195`

```cpp
Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::Tensor result, unused;
  result = at::native::_matmul_impl(unused, tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}
```

### 3.2 _matmul_impl 的维度分发

**位置**: `LinearAlgebra.cpp:2002-2180`

```cpp
static Tensor _matmul_impl(Tensor& out, const Tensor& tensor1, const Tensor& tensor2) {
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();

  // 1D @ 1D → dot()
  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? at::dot_out(out, tensor1, tensor2)
                   : tensor1.dot(tensor2);
  }

  // 2D @ 1D → mv()（矩阵向量乘）
  else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? at::mv_out(out, tensor1, tensor2)
                   : tensor1.mv(tensor2);
  }

  // 1D @ 2D → mm()（扩展后的矩阵乘）
  else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out
      ? at::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
      : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  }

  // 2D @ 2D → mm()（直接矩阵乘）
  else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1, tensor2)
                   : tensor1.mm(tensor2);
  }

  // 高维 → bmm() 或 folding 优化
  else if (should_fold(tensor1, tensor2, has_out)) {
    // 优化路径：将批量维度折叠为矩阵维度
    at::_unsafe_view(t1_folded.mm(*t2), output_shape);
  } else {
    // 批量路径
    tensor1_expanded.bmm(tensor2_expanded);
  }
}
```

---

## 4. 第二层分发：CUDA Blas 选择

### 4.1 mm_out_cuda

**位置**: `aten/src/ATen/native/cuda/Blas.cpp:668-671`

```cpp
TORCH_IMPL_FUNC(mm_out_cuda)(const Tensor& self, const Tensor& mat2,
                              const Tensor& result) {
  // mm 复用 addmm，即 C = 0*C + 1*(A @ B)
  addmm_out_cuda_impl(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
}
```

### 4.2 bmm_out_cuda

**位置**: `Blas.cpp:680-687`

```cpp
TORCH_IMPL_FUNC(bmm_out_cuda)(const Tensor& batch1, const Tensor& batch2,
                               const Tensor &result) {
  Scalar beta(0.0);
  Scalar alpha(1.0);
  {
    NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, result, batch1, batch2, beta, alpha);
  }
}
```

**关键观察**：2D 和 3D 操作都统一到更底层的实现。

### 4.3 addmm_out_cuda_impl 的决策

**位置**: `Blas.cpp:318-560`

```cpp
Tensor& addmm_out_cuda_impl(
    Tensor& result, const Tensor& self,
    const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta, const Scalar& alpha,
    Activation activation=Activation::None)
```

#### 决策 1: cublasLt 接口选择

**位置**: `Blas.cpp:336-387`

```cpp
static bool disable_addmm_cuda_lt = getDisableAddmmCudaLt();

if (!disable_addmm_cuda_lt) {
  useLtInterface =
      beta.toComplexDouble() == 1.0 &&
      self.dim() == 1 &&
      result.dim() == 2 &&
      self.sizes()[0] == mat2_sizes[1] &&
      self.is_contiguous() &&
      result.is_contiguous() &&
      (scalar_type == at::ScalarType::Double ||
       scalar_type == at::ScalarType::Float ||
       scalar_type == at::ScalarType::Half ||
       scalar_type == at::ScalarType::BFloat16) &&
      mat2_sizes[0] > 1 && mat2_sizes[1] > 1;
}
```

**判断条件**：
- CUDA 版本
- 数据类型 (float, double, half, bfloat16)
- 张量形状 (矩阵大小限制)
- 内存布局 (contiguity, strides)

#### 决策 2: 数据类型分发

**位置**: `Blas.cpp:438-514`

**当 useLtInterface=true 时**：

```cpp
if (useLtInterface) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "addmm_cuda_lt",
      [&] {
        auto tuning_ctx = at::cuda::tunable::getTuningContext();
        if (tuning_ctx->IsTunableOpEnabled()) {
          // 可调参数路径 - tunable ops
          launchTunableGemmAndBias<scalar_t>(...);
        }
        else {
          // cublasLt 路径 - gemm_and_bias
          at::cuda::blas::gemm_and_bias<scalar_t>(...);
        }
      });
}
```

**当 useLtInterface=false 时**：

```cpp
else {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "addmm_cuda",
      [&] {
        // 标准 gemm 路径
        at::cuda::blas::gemm<scalar_t>(...);
      });
}
```

---

## 5. 第三层分发：BLAS 后端选择

### 5.1 gemm_internal 的运行时分发

**位置**: `aten/src/ATen/cuda/CUDABlas.cpp:1054-1088`

```cpp
template <>
void gemm_internal<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    // cublasLt 路径 - 更新的 NVIDIA BLAS library
    gemm_internal_cublaslt<double>(CUDABLAS_GEMM_ARGS(double));
  }
#ifdef USE_ROCM
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    // AMD ROCm 路径 - Composable Kernel
    at::native::gemm_internal_ck<double>(CUDABLAS_GEMM_ARGS(double));
  }
#endif
  else {
    // 标准 cuBLAS 路径
    gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGS(double));
  }
}
```

### 5.2 BlasBackend 枚举

**位置**: `BlasBackend.h`

```cpp
enum class BlasBackend : int8_t {
    Default,    // 使用默认后端
    Cublas,     // 强制使用 cuBLAS
    Cublaslt,   // 强制使用 cuBLASLt
    Ck          // AMD Composable Kernel (ROCm only)
};
```

---

## 6. 具体 BLAS 调用

### 6.1 gemm_internal_cublas

**位置**: `CUDABlas.cpp:858-868`

```cpp
template <>
void gemm_internal_cublas<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);

  // 调用 cuBLAS SGEMM
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}
```

### 6.2 gemm_internal_cublaslt

**位置**: `CUDABlas.cpp:1295-1425`

cublasLt 是更新的 API，支持自动算法搜索：

```cpp
cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

// 设置计算描述符
CuBlasLtMatmulDesc computeDesc(...);
computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);

// 设置矩阵布局
CuBlasLtMatrixLayout Adesc(abcType, m, k, lda, transpose_mat1);
CuBlasLtMatrixLayout Bdesc(abcType, k, n, ldb, transpose_mat2);
CuBlasLtMatrixLayout Cdesc(abcType, m, n, ldc);

// 设置性能偏好
CuBlasLtMatmulPreference preference;
preference.setAttribute(
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

// 获取启发式算法选择
cublasLtMatmulHeuristicResult_t heuristicResult = {};
TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
    ltHandle,
    computeDesc.descriptor(),
    Adesc.descriptor(),
    Bdesc.descriptor(),
    Cdesc.descriptor(),
    Cdesc.descriptor(),
    preference.descriptor(),
    1,
    &heuristicResult,
    &returnedResult));

// 执行矩阵乘法
cublasStatus_t cublasStatus = cublasLtMatmul(
    ltHandle,
    computeDesc.descriptor(),
    alpha_ptr,
    mat1_ptr, Adesc.descriptor(),
    mat2_ptr, Bdesc.descriptor(),
    beta_ptr,
    result_ptr, Cdesc.descriptor(),
    result_ptr, Cdesc.descriptor(),
    &heuristicResult.algo,
    workspace.mutable_data_ptr(),
    workspaceSize,
    at::cuda::getCurrentCUDAStream());
```

### 6.3 gemm_and_bias

**位置**: `CUDABlas.cpp:1295-1425`

支持在矩阵乘后应用偏置和激活函数：

```cpp
template <>
void gemm_and_bias_cublaslt<float>(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<float> alpha_val,
    const float* mat1_ptr, int64_t mat1_ld,
    const float* mat2_ptr, int64_t mat2_ld,
    const float* bias,
    float* result_ptr, int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation)
```

**Epilogue 操作选择**：
```cpp
cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
if (activation == GEMMAndBiasActivationEpilogue::RELU) {
  epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
} else if (activation == GEMMAndBiasActivationEpilogue::GELU) {
  epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;  // CUDA 11.4+
}
```

---

## 7. 第四层分发：Tunable GEMM

### 7.1 转置配置分发

**位置**: `Blas.cpp:279-316`

当启用 autotuning 时，根据转置配置进行模板特化：

```cpp
static void launchTunableGemmAndBias(
    cublasCommonArgs &args,
    const Scalar& alpha,
    const scalar_t* bias,
    cuda::blas::GEMMAndBiasActivationEpilogue activation)
{
  bool transa_ = ((args.transa != 'n') && (args.transa != 'N'));
  bool transb_ = ((args.transb != 'n') && (args.transb != 'N'));

  if (transa_ && transb_) {
    // T x T
    static at::cuda::tunable::GemmAndBiasTunableOp<
        scalar_t, at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::T> gemm{};
    gemm(&params);
  }
  else if (transa_ && !transb_) {
    // T x N
    static at::cuda::tunable::GemmAndBiasTunableOp<
        scalar_t, at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::N> gemm{};
    gemm(&params);
  }
  else if (!transa_ && transb_) {
    // N x T
    static at::cuda::tunable::GemmAndBiasTunableOp<
        scalar_t, at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::T> gemm{};
    gemm(&params);
  }
  else {
    // N x N
    static at::cuda::tunable::GemmAndBiasTunableOp<
        scalar_t, at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::N> gemm{};
    gemm(&params);
  }
}
```

这是**编译期决策**（模板特化），在编译时为每种转置组合生成不同代码。

---

## 8. 分发链总结

```
matmul(2D @ 2D)
  ↓ [第一层：维度分析 - 运行时]
mm()
  ↓ [native_functions dispatch]
mm_out_cuda()
  ↓ [复用 addmm_out_cuda_impl]
addmm_out_cuda_impl()
  ↓ [第二层：cublasLt vs cuBLAS - 运行时]
  │
  ├─ useLtInterface=true
  │   ↓ [AT_DISPATCH_FLOATING_TYPES_AND2 - 编译期]
  │   ↓ [第三层：tunable ops? - 运行时]
  │   │
  │   ├─ IsTunableOpEnabled()=true
  │   │   ↓ launchTunableGemmAndBias()
  │   │   ↓ [第四层：转置配置 - 编译期模板]
  │   │   ↓ GemmAndBiasTunableOp<scalar_t, TransA, TransB>
  │   │
  │   └─ IsTunableOpEnabled()=false
  │       ↓ gemm_and_bias_cublaslt()
  │       ↓ cublasLtMatmulAlgoGetHeuristic (启发式)
  │       ↓ cublasLtMatmul()
  │
  └─ useLtInterface=false
      ↓ [AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 - 编译期]
      ↓ gemm<scalar_t>()
      ↓ [第三层：BlasBackend - 运行时/全局设置]
      │
      ├─ Cublaslt → gemm_internal_cublaslt()
      │              → cublasLtMatmul()
      │
      ├─ Ck (ROCm) → gemm_internal_ck()
      │
      └─ Default/Cublas → gemm_internal_cublas()
                          → cublasSgemm()/cublasDgemm()
```

---

## 9. 分发决策因素总结

| 层级 | 决策因素 | 类型 |
|------|---------|------|
| 维度分析 | 输入张量的 ndim (1D/2D/3D/更高) | 运行时 |
| cublasLt 选择 | dtype, contiguity, 矩阵大小, beta 值 | 运行时 |
| 数据类型分发 | `AT_DISPATCH_FLOATING_TYPES_AND2(...)` | 编译期模板 |
| 后端选择 | `globalContext().blasPreferredBackend()` | 全局设置 |
| 转置配置 | `GemmAndBiasTunableOp<T, TransA, TransB>` | 编译期模板 |
| 算法选择 | `cublasLtMatmulAlgoGetHeuristic` | 运行时启发式 |

---

## 10. 使用的 CUDA 库

| 库 | API | 用途 |
|----|-----|------|
| **cuBLAS** | `cublasSgemm()`, `cublasDgemm()` | 传统矩阵乘法 |
| **cuBLASLt** | `cublasLtMatmul()` | 高性能、可调参数、支持 epilogue |
| **Composable Kernel** | `gemm_internal_ck()` | AMD ROCm 优化库 |

---

## 11. 关键代码位置

| 组件 | 文件 | 行号 |
|------|------|------|
| matmul native 定义 | `native_functions.yaml` | 3775-3789 |
| mm native 定义 | `native_functions.yaml` | 4133-4150 |
| matmul 顶层 API | `LinearAlgebra.cpp` | 2182-2195 |
| _matmul_impl 维度分发 | `LinearAlgebra.cpp` | 2002-2180 |
| mm_out_cuda | `cuda/Blas.cpp` | 668-671 |
| addmm_out_cuda_impl | `cuda/Blas.cpp` | 318-560 |
| addmv_out_cuda | `cuda/Blas.cpp` | 823-876 |
| gemm_internal 后端分发 | `CUDABlas.cpp` | 1054-1088 |
| gemm_internal_cublas | `CUDABlas.cpp` | 858-868 |
| gemm_internal_cublaslt | `CUDABlas.cpp` | 1295-1425 |
| launchTunableGemmAndBias | `cuda/Blas.cpp` | 279-316 |
| BlasBackend 枚举 | `BlasBackend.h` | - |
