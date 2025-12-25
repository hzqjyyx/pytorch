# CPUBlas 文件功能理解

这两个文件实现了 PyTorch CPU 端的 BLAS (Basic Linear Algebra Subprograms) 操作接口层，主要负责矩阵乘法等基础线性代数运算。

## 核心架构

**分层调度机制**：实现了一个三层调度系统
1. 顶层接口：提供类型安全的重载函数
2. 库选择层：根据条件选择使用外部 BLAS 库（MKL、OpenBLAS、Apple Accelerate）或自定义实现
3. 后备实现：通过 `gemm_stub`/`axpy_stub`/`copy_stub` 提供纯 CPU 实现

## 主要功能模块

### 1. GEMM (General Matrix Multiply)

实现 `C = alpha * op(A) * op(B) + beta * C` 的矩阵乘法运算。

**支持的数据类型**：
- 标准浮点：`double`, `float`
- 低精度浮点：`BFloat16`, `Half`
- 复数：`c10::complex<double>`, `c10::complex<float>`
- 整数：`int64_t`

**调度逻辑示例**（以 `float` 版本为例，CPUBlas.cpp:182-228）：
```cpp
void gemm(float版本) {
  normalize_last_dims();  // 标准化维度参数
  
  #if AT_MKLDNN_ENABLED()
    if (mkldnn_bf32_gemm(...)) return;  // 优先尝试 oneDNN BF32 优化
  #endif
  
  #if AT_BUILD_WITH_BLAS()
    if (use_blas_gemm(...)) {  // 检查维度是否在 INT_MAX 范围内
      sgemm_(...);  // 调用外部 BLAS 库
      return;
    }
  #endif
  
  gemm_stub(...);  // 后备：使用 PyTorch 自己的实现
}
```

**特殊处理**：
- **BFloat16/Half 混合精度**：支持输入 BF16/FP16 但输出 FP32 的计算路径（CPUBlas.cpp:402-487）
  - 优先使用 MKL 的 `mkl_gemm_bf16bf16f32` 专用函数
  - 后备方案：先以 BF16 计算得到中间结果，再转换为 FP32 并累加
  
- **特定 GEMV 优化绕过**（CPUBlas.cpp:354-362）：
  ```cpp
  // 对于 BF16，当满足以下条件时绕过 oneDNN：
  // 1. CPU 不支持 AVX512_BF16
  // 2. 是转置矩阵乘向量 (transa=T, transb=N, n=1, alpha=1.0)
  // 原因：此场景下 PyTorch 自己的实现更快
  ```

### 2. 批量 GEMM

**两种接口**：
- `gemm_batched`：指针数组形式，`A[]`, `B[]`, `C[]` 分别指向各批次矩阵
- `gemm_batched_with_stride`：单一连续内存，通过 stride 访问各批次

**优化策略**（CPUBlas.cpp:571-596）：
```cpp
if (batch_size == 1) {
  return gemm(...);  // 单批次直接调用普通 gemm
}

if constexpr (AT_MKL_ENABLED() && is_blas_library_type) {
  if (use_blas_gemm(...)) {
    gemm_batched_mkl_impl(...);  // 使用 MKL 批量接口
  } else {
    gemm_batched_generic(...);  // 逐个调用 gemm
  }
}
```

### 3. BRGEMM (Batch-Reduce GEMM)

基于 oneDNN ukernel 的高性能矩阵乘法，主要用于加速深度学习场景。

**核心公式**：`C = SUM(A[i] × B[i]) + C` （目前仅支持 batch_size=1）

**VNNI 布局支持**（CPUBlas.cpp:1240-1300）：
- VNNI (Vector Neural Network Instructions) 是 Intel CPU 的向量化指令集
- BFloat16/Half/Int8 可使用 VNNI 布局以提升性能
- Float 不支持 VNNI（CPUBlas.cpp:1220-1221）

**版本兼容处理**（CPUBlas.cpp:52-59）：
```cpp
// oneDNN 3.6.x 更改了 ukernel API：
// - brgemm_pack_B → transform
// - beta 设置 → set_add_C
#if IDEEP_VERSION == 3.5
  #define ONEDNN_UKERNEL_1  // 旧 API
#elif IDEEP_VERSION >= 3.6
  #define ONEDNN_UKERNEL_2  // 新 API
#endif
```

**缓存机制**（CPUBlas.cpp:994-1013）：
```cpp
struct KernelCache {
  static inline std::shared_ptr<value_t>&& fetch_or_create(...) {
    static thread_local kstore_t cache_kernels;  // 线程局部缓存
    if (cache.find(key) != cache.end()) {
      return cache[key];
    }
    cache.insert({key, callback()});
    return cache[key];
  }
};
```

**设备能力检测**（CPUBlas.cpp:1129-1150）：
- FP16：需要 AVX512_CORE_FP16
- FP32：需要 AVX2
- BF16：需要 AVX512_CORE
- U8：需要 AVX512_CORE_AMX
- S8：需要 AVX512_CORE_VNNI

### 4. Pack 操作

用于将矩阵 B 重排为 VNNI 布局以提升 BRGEMM 性能（CPUBlas.h:266-277）。

**支持场景**（CPUBlas.cpp:1188-1203）：
- BFloat16：需要 AVX512_CORE_AMX
- Half：需要 AVX512_CORE_AMX_FP16
- Int8 (Byte/Char)：需要 AVX512_CORE_AMX

### 5. AXPY 和 COPY

**AXPY**：`y = a*x + y`（CPUBlas.cpp:677-775）
- 支持 `double`, `float`, `complex<double>`, `complex<float>`
- 单元素特殊处理：强制 `incx=1, incy=1`

**COPY**：`y = x`（CPUBlas.cpp:779-873）
- 相同的类型支持和单元素优化

## 关键辅助功能

### normalize_last_dims (CPUBlas.cpp:70-93)

根据矩阵转置类型和维度自动修正 leading dimension，确保内存访问的正确性：
```cpp
if (n == 1) ldc = m;  // 结果是向量
if (transa != NoTranspose) {
  if (m == 1) lda = k;
} else if (k == 1) {
  lda = m;
}
// 类似逻辑处理 ldb
```

### use_blas_gemm (CPUBlas.cpp:98-110)

检查是否可以安全调用外部 BLAS 库：
- 所有维度参数必须 ≤ INT_MAX（BLAS 接口限制）
- leading dimension 必须满足 BLAS 规范要求

### to_blas 转换

将 PyTorch 的 `TransposeType` 枚举转换为：
- Fortran BLAS：字符 `'N'/'T'/'C'`
- Apple Accelerate：`CBLAS_TRANSPOSE` 枚举
- FBGEMM：`matrix_op_t` 枚举

## 编译配置相关

**条件编译宏**：
- `AT_BUILD_WITH_BLAS()`：是否链接外部 BLAS 库
- `AT_MKL_ENABLED()`：是否使用 Intel MKL
- `AT_MKLDNN_ENABLED()`：是否启用 oneDNN
- `C10_IOS`：iOS 平台（使用 Accelerate 框架）
- `USE_FBGEMM`：使用 FBGEMM 库（用于 int64 gemm）
- `BLAS_HAS_SBGEMM`：BLAS 库是否支持 BFloat16 gemm

**外部符号声明**（CPUBlas.cpp:19-40）：
非 iOS 平台需要手动声明 Fortran BLAS 函数（带下划线后缀）：
```cpp
extern "C" void dgemm_(...);
extern "C" void sgemm_(...);
// 等等
```

---

**其他内容**：
- ROCm 相关：文件中未涉及 ROCm 特定代码
- Backward 相关：这些是前向计算的底层实现，不直接处理反向传播
