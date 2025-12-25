这个文件实现了基于 cuBLAS 的批量线性代数运算，作为 PyTorch CUDA 后端的一部分。

## 核心功能

**1. QR 分解 (geqrf_batched_cublas)**
- 使用 cuBLAS 的 `geqrfBatched` 进行批量 QR 分解
- 输入矩阵 A 分解为正交矩阵 Q 和上三角矩阵 R
- 返回的 tau 参数用于重构 Q 矩阵

**2. LU 分解 (lu_factor_batched_cublas)**
- 调用 `getrfBatched` 进行批量 LU 分解（带部分主元）
- 将矩阵分解为下三角 L 和上三角 U
- 输出 pivots（主元索引）和 infos（状态信息）

**3. LU 求解线性方程组 (lu_solve_batched_cublas)**
- 使用 `getrsBatched` 基于 LU 分解求解 Ax=b
- 支持转置选项（普通/转置/共轭转置）
- 需要预先计算的 LU 分解和主元信息

**4. 三角求解器 (triangular_solve_cublas / triangular_solve_batched_cublas)**
- 求解三角线性系统 AX=B 或 XA=B
- 支持上/下三角、左/右乘、转置、单位对角线等选项
- 批量版本使用 `trsmBatched`，非批量版本循环调用 `trsm`
- 包含 CUDA < 12.1 的 bug 修复（batch_size > 524280 时分块处理）

**5. 最小二乘法 (gels_batched_cublas)**
- 使用 `gelsBatched` 求解超定方程组（m ≥ n）
- 找到使 ||Ax - b||₂ 最小的解
- cuBLAS 要求 nrhs > 0，需要提前返回处理
- 需要显式广播矩阵 A 以避免重叠问题

## 关键技术细节

**设备指针数组生成 (get_device_pointers)**
- cuBLAS 批量 API 要求输入为"指向设备矩阵的设备指针数组"
- 使用 `arange` 生成包含各矩阵起始地址的 Tensor
- 处理 batch_size=0 和 stride=0 的边界情况

**类型转换 (to_cublas)**
- 将 PyTorch 的 TransposeType 转换为 cuBLAS 枚举
- NoTranspose → CUBLAS_OP_N
- Transpose → CUBLAS_OP_T  
- ConjTranspose → CUBLAS_OP_C

**错误处理**
- 使用 `cuda_int_cast` 确保维度符合 cuBLAS 的 int 要求
- 检查 info 返回值确认 API 调用成功

---

• **ROCm 相关**: 此文件专门处理 cuBLAS API，对应的 cuSOLVER API 在 BatchLinearAlgebraLib.cpp 中，分离是为了支持 ROCm 的 hipify 构建流程

• **Backward 相关**: 文件中没有显式的反向传播实现，梯度计算由上层 autograd 机制通过这些前向函数的雅可比矩阵处理
