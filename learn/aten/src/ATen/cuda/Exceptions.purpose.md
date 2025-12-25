# ATen/cuda/Exceptions 文件功能分析

## 核心功能

这两个文件为 PyTorch 的 CUDA 操作提供了统一的错误处理机制，将各种 CUDA 库的错误码转换为可读的错误信息，并通过宏定义实现错误检查。

## Exceptions.h 主要内容

### 1. CuDNN 错误处理

**自定义异常类型**（第 24-26 行）：
```cpp
class CuDNNError : public c10::Error {
  using Error::Error;
};
```

**错误检查宏**：
- `AT_CUDNN_CHECK`（第 42-56 行）：检查 cuDNN 操作返回状态，失败时抛出 `CuDNNError`，针对 `CUDNN_STATUS_NOT_SUPPORTED` 提供特殊提示（可能是非连续输入导致）
- `AT_CUDNN_FRONTEND_CHECK`（第 30-37 行）：用于 cuDNN Frontend API，检查返回对象的 `is_good()` 状态

### 2. cuBLAS 错误处理

**错误转换函数**（第 59 行）：
```cpp
const char* _cublasGetErrorEnum(cublasStatus_t error);
```

**错误检查宏** `TORCH_CUDABLAS_CHECK`（第 62-69 行）：
- 执行 cuBLAS 操作并检查返回状态
- 失败时调用 `_cublasGetErrorEnum` 将错误码转为字符串
- 包含调用表达式的完整信息

### 3. cuSPARSE 错误处理

**错误检查宏** `TORCH_CUDASPARSE_CHECK`（第 73-80 行）：
- 类似 cuBLAS 的检查机制
- 使用 `cusparseGetErrorString` 获取错误描述

### 4. cuSOLVER 错误处理

**错误转换函数**（第 112 行）：
```cpp
const char* cusolverGetErrorMessage(cusolverStatus_t status);
```

**特殊错误处理** `TORCH_CUSOLVER_CHECK`（第 122-144 行）：
- **版本相关处理**：
  - CUDA < 11.5：检测 `CUSOLVER_STATUS_EXECUTION_FAILED`（输入含 NaN）
  - CUDA >= 11.5：检测 `CUSOLVER_STATUS_INVALID_VALUE`（输入含 NaN）
- **用户友好提示**：
  - 提供 NaN 输入的明确说明
  - 包含 `_cusolver_backend_suggestion`（第 114-118 行），建议使用 `torch.backends.cuda.preferred_linalg_library()` 切换后端

### 5. cuDSS 错误处理（条件编译）

**错误检查宏** `TORCH_CUDSS_CHECK`（第 87-107 行）：
- 仅在定义 `USE_CUDSS` 时启用
- 对 `CUDSS_STATUS_EXECUTION_FAILED` 使用 `TORCH_CHECK_LINALG`（线性代数专用检查）
- 提供 NaN 输入提示

### 6. CUDA Driver API 错误处理

**宏定义** `AT_CUDA_DRIVER_CHECK`（第 188-201 行）：
- 检查 CUDA Driver API 调用（返回 `CUresult`）
- 动态获取错误字符串：通过 `at::globalContext().getNVRTC().cuGetErrorString()` 获取，因为 NVRTC 是动态加载的

### 7. NVRTC 错误处理

**宏定义** `AT_CUDA_NVRTC_CHECK`（第 224-234 行）：
- 检查 NVRTC（NVIDIA Runtime Compilation）操作
- **Bug 修复**：CUDA 10 中错误码 7（`NVRTC_ERROR_BUILTIN_OPERATION_FAILURE`）会错误返回 "NVRTC unknown error"，宏手动修正为正确的错误信息

## Exceptions.cpp 主要内容

### 1. cuBLAS 错误码转换（第 9-43 行）

```cpp
const char* _cublasGetErrorEnum(cublasStatus_t error)
```

通过一系列 if 语句将 cuBLAS 错误码映射到字符串：
- `CUBLAS_STATUS_SUCCESS`
- `CUBLAS_STATUS_NOT_INITIALIZED`
- `CUBLAS_STATUS_ALLOC_FAILED`
- `CUBLAS_STATUS_INVALID_VALUE`
- `CUBLAS_STATUS_ARCH_MISMATCH`
- `CUBLAS_STATUS_MAPPING_ERROR`
- `CUBLAS_STATUS_EXECUTION_FAILED`
- `CUBLAS_STATUS_INTERNAL_ERROR`
- `CUBLAS_STATUS_NOT_SUPPORTED`
- `CUBLAS_STATUS_LICENSE_ERROR`（条件编译）

### 2. cuSOLVER 错误码转换（第 50-62 行）

```cpp
const char* cusolverGetErrorMessage(cusolverStatus_t status)
```

使用 switch-case 转换错误码：
- `CUSOLVER_STATUS_SUCCESS`
- `CUSOLVER_STATUS_NOT_INITIALIZED`
- `CUSOLVER_STATUS_ALLOC_FAILED`
- `CUSOLVER_STATUS_INVALID_VALUE`
- `CUSOLVER_STATUS_ARCH_MISMATCH`
- `CUSOLVER_STATUS_EXECUTION_FAILED`
- `CUSOLVER_STATUS_INTERNAL_ERROR`
- `CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`

### 3. cuDSS 错误码转换（第 91-102 行）

```cpp
const char* cudssGetErrorMessage(cudssStatus_t status)
```

条件编译（`USE_CUDSS`），转换 cuDSS 库错误码。

## 设计特点

1. **统一的错误处理模式**：所有宏都遵循 "执行-检查-转换-抛出" 模式
2. **信息完整性**：错误信息包含库名称、错误描述、调用表达式
3. **用户友好**：针对常见错误（如 NaN 输入）提供明确的诊断和解决建议
4. **版本兼容性**：处理不同 CUDA 版本的 API 差异（如 cuSOLVER 的 NaN 检测）
5. **平台适配**：通过条件编译支持 CUDA 和 ROCm

---

**ROCm 相关内容**：
- `hipsolverGetErrorMessage` 函数用于 AMD ROCm 平台的 HIP solver 错误转换
- `TORCH_CUSOLVER_CHECK` 在 ROCm 下使用 `hipsolver` 替代实现
- ROCm 的 Driver API 检查不支持动态获取错误字符串，直接输出错误码整数
