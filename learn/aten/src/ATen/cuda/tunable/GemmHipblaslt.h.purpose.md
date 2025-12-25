# GemmHipblaslt.h 核心功能分析

这个文件为 PyTorch 提供了基于 hipBLASLt 的可调优 GEMM（通用矩阵乘法）操作实现。

## 主要组件

### 1. 数据类型映射系统

将 PyTorch 的数据类型映射到 HIP 运行时类型：

```cpp
template <typename T>
constexpr hipDataType HipDataTypeFor();
```

支持类型：float、Half、BFloat16、double、Float8 系列（e4m3fnuz、e5m2fnuz 等）

### 2. 参数提取器模板函数族

通过模板特化从不同的 GEMM 参数结构体中提取配置：

- **批次信息**：`GetBatchFromParams` - 从 GemmStridedBatchedParams 提取批次大小，其他返回 1
- **步长信息**：`GetStrideA/B/CFromParams` - 提取矩阵的批次步长
- **缩放因子**：`GetAlphaFromParams`、`GetBetaFromParams` - ScaledGemmParams 中 alpha 固定为 1.0
- **量化缩放指针**：`GetA/B/DScalePointerFromParams` - 仅 ScaledGemmParams 提供
- **偏置配置**：`GetBiasPointerFromParams`、`GetBiasTypeFromParams` - GemmAndBiasParams 和 ScaledGemmParams 支持
- **激活函数**：`GetActivationFromParams` - 提取 RELU/GELU 等后处理操作

### 3. HipblasltGemmOp 核心执行类

模板类 `HipblasltGemmOp<AT, BT, CT, ALayout, BLayout, ParamsT>` 实现了 GEMM 操作的可调用接口：

**主要流程**（Call 方法）：

1. **矩阵布局设置**
   - 根据转置标志（transa/transb）创建 hipBLASLt 矩阵布局对象
   - 为批次 GEMM 配置步长和批次计数

2. **描述符配置**
   - 创建 `HipBlasLtMatmulDescriptor` 设置计算类型和转置操作
   - 为量化 GEMM 设置缩放指针（`A_SCALE_POINTER`、`B_SCALE_POINTER`、`D_SCALE_POINTER`）
   - 配置偏置和激活函数融合（EPILOGUE_BIAS、EPILOGUE_RELU_BIAS、EPILOGUE_GELU_BIAS）

3. **算法验证**
   - 使用 `hipblaslt_ext::matmulIsAlgoSupported` 检查算法是否支持
   - 验证所需工作空间大小是否在限制内

4. **执行计算**
   - 分配工作空间（通过 CUDA 缓存分配器）
   - 调用 `hipblasLtMatmul` 执行矩阵乘法
   - 清理资源

**返回值**：
- `OK` - 成功
- `FAIL` - 算法不支持或工作空间超限

### 4. 算法枚举函数

`GetHipBlasLtTypeStringAndOps()` 及其特化版本：

- 创建临时 hipBLASLt 句柄
- 调用 `hipblaslt_ext::getAllAlgos` 获取所有可用算法
- 为每个算法创建 `HipblasltGemmOp` 实例
- 返回 `<算法名称, Callable对象>` 的向量

四个便捷包装函数：
- `GetHipBlasLtGemmTypeStringAndOps` - 标准 GEMM
- `GetHipBlasLtGemmAndBiasTypeStringAndOps` - GEMM + 偏置
- `GetHipBlasLtGemmStridedBatchedTypeStringAndOps` - 批次 GEMM
- `GetHipBlasLtScaledGemmTypeStringAndOps` - 量化 GEMM

### 5. 辅助工具

**布局转换**：
```cpp
hipblasOperation_t MapLayoutToHipBlasLt(BlasOp layout)
```
将 PyTorch 的 BlasOp 枚举转换为 hipBLAS 操作类型

**工作空间大小配置**：
```cpp
GetHipblasltWorkspaceSize()
```
从环境变量 `HIPBLASLT_WORKSPACE_SIZE` 读取，默认 76MB

**RAII 描述符管理**：
```cpp
HipBlasLtDescriptor<T, destructor>
HipBlasLtMatmulDescriptor
```
使用智能指针自动管理 hipBLASLt 资源生命周期

## 设计特点

- **模板元编程**：通过编译时类型推导和特化实现零开销抽象
- **统一接口**：不同 GEMM 变体（标准/批次/量化/偏置）共用同一套实现框架
- **可调优性**：支持枚举所有算法供运行时选择最优配置
- **内存管理**：利用 PyTorch 的 CUDA 缓存分配器管理工作空间
- **融合操作**：支持偏置加法和激活函数融合以提升性能

---

**ROCm/Backward 相关忽略项**：
- ROCm 6.3 版本条件编译（Float8_e4m3fn、Float8_e5m2 枚举值兼容）
- `HIPBLASLT_VEC_EXT` 宏保护的行向量缩放支持
- hipBLAS 向后兼容的操作码映射函数
