- **文件目的**：为 PyTorch 的 GEMM（通用矩阵乘法）操作提供 ROCblas 后端实现

- **数据类型映射**：
  - `RocBlasDataTypeFor<T>()`：将 C++ 类型（float, double, Half, BFloat16, complex）映射到 ROCblas 数据类型
  - `RocBlasComputeTypeFor<T>()`：返回计算用的数据类型（Half 和 BFloat16 使用 FP32 计算以获得更好性能）

- **类型转换工具**：
  - `DoCastForHalfOrBfloat16()`：将 Half/BFloat16 转换为 float，因为 ROCblas API 要求 alpha/beta 参数匹配计算类型

- **操作转换**：
  - `_rocblasOpFromChar()`：将字符 'n'/'t'/'c' 转换为对应的 ROCblas 操作枚举值

- **核心类**：
  - `RocblasGemmOp<T>`：单个 GEMM 操作的可调用对象，调用 `rocblas_gemm_ex()`
  - `RocblasGemmStridedBatchedOp<T>`：批量 GEMM 操作的可调用对象，调用 `rocblas_gemm_strided_batched_ex()`

- **可调用对象生成**：
  - `GetRocBlasGemmTypeStringAndOps()`：枚举所有可用的 GEMM 解决方案，为每个返回一个字符串名称和对应的可调用对象
  - `GetRocBlasGemmStridedBatchedTypeStringAndOps()`：类似，但针对分层批量 GEMM，并对解决方案排序以保证确定性

- **错误检查**：
  - `TORCH_ROCBLAS_CHECK` 宏：在每次 ROCblas 调用后检查状态，失败时输出错误信息
