# GemmCommon.h 主要功能分析

这个文件为 PyTorch 的 TunableOp 框架提供 GEMM（通用矩阵乘法）操作的通用定义和参数结构。

## 核心组件

### 1. 类型映射系统

**BLAS 类型名称映射** (lines 48-102)：
- 将 C++ 类型映射到 BLAS 字符串表示
- `BLASTypeName<T>()` 模板函数处理各种数值类型：
  - 标准浮点：`float` → `"f32_r"`, `double` → `"f64_r"`
  - 半精度：`Half` → `"f16_r"`, `BFloat16` → `"bf16_r"`
  - 8位浮点：`Float8_e4m3fn` → `"f8_r"`, `Float8_e5m2` → `"bf8_r"` 等
  - 复数类型：`complex<float>` → `"f32_r"`, `complex<double>` → `"f64_r"`

**计算类型推导** (lines 154-213)：
- `ComputeTypeFor<T>()` 决定实际运算时使用的精度
- 关键逻辑：
  - `float` 类型根据 `allowTF32CuBLAS()` 返回 `"f32_r"` 或 `"xf32_r"`
  - `Half/BFloat16` 使用 `f32` 进行内部计算
  - 8位浮点类型也提升到 `f32` 计算

### 2. 参数结构体

文件定义了四个主要的参数结构，都继承自 `OpParams`：

#### `GemmParams<T>` (lines 281-375) - 基础 GEMM
```
C = alpha * op(A) * op(B) + beta * C
```
- 核心字段：矩阵维度 (m,n,k)、转置标志 (transa,transb)、leading dimensions (lda,ldb,ldc)、缩放因子 (alpha,beta)
- 关键方法：
  - `Signature()`: 生成唯一标识字符串 `"{trans}{trans}_{m}_{n}_{k}_ld_{lda}_{ldb}_{ldc}"`
  - `BLASSignature()`: 生成完整的 BLAS 调用签名（YAML 格式）
  - `DeepCopy()`: 分配 GPU 内存并复制参数（用于性能测试）
  - `GetSize*()`: 计算所需内存大小，考虑 stride 和 dense layout

#### `GemmAndBiasParams<T>` (lines 377-470) - GEMM + Bias + 激活
```
C = activation(alpha * op(A) * op(B) + bias)
```
- 扩展字段：`bias` 指针、`activation` 枚举（None/RELU/GELU）
- 结构与 `GemmParams` 类似，但没有 `beta` 参数

#### `GemmStridedBatchedParams<T>` (lines 472-570) - 批量 GEMM
```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i], for i in [0, batch)
```
- 额外字段：`stride_a/b/c`（批次间步长）、`batch`（批次数）
- `GetSize*()` 方法考虑批次维度

#### `ScaledGemmParams<T>` (lines 572-683) - 缩放 GEMM
```
C = op(A) * scale_A * op(B) * scale_B + bias
```
- 支持混合精度：A, B, C 可以是不同的数据类型
- 特殊字段：
  - 每个矩阵的独立 dtype 和 scale 指针
  - `use_fast_accum`: 快速累加标志
  - `use_rowwise`: 行级缩放标志
  - `amax_ptr`: 输出绝对最大值（用于量化）
- 主要用于 FP8 等低精度训练场景

### 3. 通用功能

**数值验证** (lines 245-272, `detail::NumericalCheck`):
- 将两个结果转换为 float 后比较
- 测试多个容差组合（atol/rtol 从 1e-1 到 1e-5）
- 用于验证不同 GEMM 实现的正确性

**格式化工具** (lines 217-241):
- `to_string_opmath<T>()`: 处理复数和实数的格式化
- `to_string_epilogue()`: 将激活函数枚举转为字符串

**内存管理模式**:
- 所有参数结构支持 `duplicate_inputs` 标志
- 启用时在 DeepCopy 中也复制输入矩阵 A, B（而非仅输出 C）
- 用于隔离测试环境，避免输入被污染

## 设计意图

1. **TunableOp 集成**: 通过 `Signature()` 生成唯一键，用于缓存最优实现选择
2. **多后端支持**: 参数格式兼容多种 BLAS 库（虽然文件在 cuda/ 目录下）
3. **性能测试**: `DeepCopy` + `NumericalCheck` 支持在不影响原数据的情况下测试不同实现
4. **类型安全**: 模板化设计确保编译期类型检查

---

**ROCm 相关**：
- 文件头注释提到源自 ONNX Runtime 并由 AMD 适配到 PyTorch
- `BLASTypeName` 注释引用 hipBLASLt 的类型定义
- `ComputeTypeFor` 注释提到兼容 ROCBLAS 和 hipBLASLt

**向后兼容/遗留**：
- `BlasOp` 枚举（lines 32-46）定义但在参数结构中未直接使用，可能为向后兼容保留
- 所有参数结构的 `duplicate_inputs_` 字段为私有，仅在内部管理生命周期
