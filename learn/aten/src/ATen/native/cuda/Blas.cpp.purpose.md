# aten/src/ATen/native/cuda/Blas.cpp 主要功能分析

这个文件实现了 PyTorch CUDA 后端的 BLAS (Basic Linear Algebra Subprograms) 操作，主要处理矩阵乘法相关的核心运算。

## 核心矩阵布局转换机制

**问题背景：** PyTorch 使用行主序（row-major），而 CUBLAS 期望列主序（column-major）。

**解决方案：** 利用数学恒等式 `(A × B)^T = B^T × A^T`

- 当计算 `C = A × B` (都是行主序) 时
- 实际调用 CUBLAS 计算 `B^T × A^T`
- CUBLAS 输出列主序结果，在 PyTorch 中解释为行主序，恰好就是期望的 `C`

**实现细节：**
- `prepare_matrix_for_cublas()`: 检查张量的步长（stride）模式，决定是否需要转置标志
- `cublasCommonArgs` 结构体: 统一处理矩阵准备、转置标志、leading dimension 计算
- 自动处理 conjugate（共轭）情况

## 主要函数实现

### 1. addmm_out_cuda_impl (aten/src/ATen/native/cuda/Blas.cpp:318)
计算 `result = beta * self + alpha * (mat1 @ mat2)`

**优化路径选择：**
- **cublasLt 路径** (CUDA >= 11.4): 
  - 条件：bias 是 1D、beta=1.0、结果连续、矩阵大小 > 1
  - 支持融合 bias 加法和激活函数（RELU/GELU）
  - 可选 TunableOp 自动调优
  
- **标准 cublas 路径**: 
  - 回退选项，分离执行 gemm 和激活函数

**特殊情况处理：**
- 空矩阵：beta=0 时直接清零，否则按 beta 缩放
- 矩阵维度检查和 GPU 设备一致性验证

### 2. baddbmm_out_cuda_impl (aten/src/ATen/native/cuda/Blas.cpp:574)
批量矩阵乘法：`result = beta * self + alpha * (batch1 @ batch2)`

- 使用 `prepare_batch_matrix_for_cublas()` 处理批次维度
- batch=1 时优化为普通 gemm
- batch>1 时调用 `bgemm` (strided batched gemm)

### 3. dot_cuda / vdot_cuda (aten/src/ATen/native/cuda/Blas.cpp:725, 774)
向量点积

- `dot`: 标准点积
- `vdot`: 复数向量点积（第一个向量取共轭）
- 设置 `CUBLAS_POINTER_MODE_DEVICE` 使结果直接写入 GPU

### 4. addmv_out_cuda (aten/src/ATen/native/cuda/Blas.cpp:823)
矩阵-向量乘法：`result = beta * self + alpha * (mat @ vec)`

- 根据矩阵步长选择 'n' 或 't' 转置标志
- 处理向量步长为 0 的边界情况

### 5. _int_mm_cuda (aten/src/ATen/native/cuda/Blas.cpp:925)
INT8 矩阵乘法

- 需要 CUDA >= 11.7
- 输入维度必须是 8 的倍数
- 调用 `at::cuda::blas::int8_gemm()`

### 6. _scaled_mm_cuda (aten/src/ATen/native/cuda/Blas.cpp:1447)
FP8 缩放矩阵乘法

**支持的缩放类型 (get_scaling_type):**
- **TensorWise**: scale_a 和 scale_b 都是标量
- **RowWise**: scale_a 是 (M, 1), scale_b 是 (1, N)
- **BlockWise**: 使用 Float8_e8m0fnu 类型，128×32 块

**约束条件：**
- 仅支持 CUDA 计算能力 >= 8.9 或 9.0
- 矩阵维度必须是 16 的倍数
- mat1 必须行主序，mat2 必须列主序
- 不支持两个 Float8_e5m2 矩阵相乘

**特殊路径：**
- RowWise 缩放调用 CUTLASS kernel (`f8f8bf16_rowwise`)
- 其他调用 `at::cuda::blas::scaled_gemm()`

### 7. _scaled_grouped_mm_cuda (aten/src/ATen/native/cuda/Blas.cpp:1461)
分组矩阵乘法（用于 MoE 模型）

- 支持 2D×3D 或 3D×2D 矩阵组合
- 需要 offsets 张量指定分组边界
- 仅支持 Float8_e4m3fn + BF16 输出

## 关键数据结构

### cublasCommonArgs (aten/src/ATen/native/cuda/Blas.cpp:131)
```cpp
struct cublasCommonArgs {
  char transa, transb;           // 转置标志 ('n'/'t'/'c')
  int64_t m, n, k;               // 矩阵维度
  int64_t lda, ldb, result_ld;   // leading dimensions
  c10::MaybeOwned<Tensor> mata, matb, result;  // 准备好的张量
  void* scale_mata_ptr;          // FP8 缩放指针
  // ...
}
```

自动处理：
- 矩阵准备和可能的克隆
- 转置标志根据 `transpose_result` 自动翻转
- FP8 缩放指针根据转置情况正确映射

## 性能优化技术

1. **避免不必要的拷贝**: 
   - `c10::MaybeOwned` 智能指针仅在必要时克隆
   - 步长检查避免连续张量的拷贝

2. **TunableOp 框架**:
   - 运行时自动选择最快的 kernel 实现
   - 支持 gemm_and_bias、scaled_gemm

3. **Epilogue Fusion**:
   - GELU/RELU 融合到 gemm kernel（需要 CUDA >= 11.8）
   - Bias 加法融合

4. **零张量优化**:
   - `_is_zerotensor()` 快速返回零张量

---

## 简要提及的其他内容

**ROCm 相关：**
- hipBLASLt 架构检测（gfx90a, gfx942, gfx1200 等）
- FNUZ 浮点格式特殊处理
- 部分功能受限（如不支持 grouped gemm）

**Backward 相关内容：**
- 文件中未直接包含反向传播实现
- 仅提供前向 BLAS 操作，梯度计算在其他文件处理
