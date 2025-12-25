这个文件定义了用于GEMM（通用矩阵乘法）操作的可调优算法框架。核心内容：

**DefaultGemmOp** (30-43行)
- 包装`at::cuda::blas::gemm_internal`的默认实现
- 处理基本矩阵乘法参数：transa、transb、m、n、k、alpha、beta等

**DefaultGemmAndBiasOp** (50-65行)
- 扩展GEMM支持偏置项和激活函数
- 调用`at::cuda::blas::gemm_and_bias`

**DefaultGemmStridedBatchedOp** (68-82行)
- 批量GEMM操作的默认实现
- 处理stride参数进行矩阵批量计算

**DefaultScaledGemmOp** (85-114行)
- 支持缩放的GEMM操作
- 涉及dtype转换、scale指针、rowwise等参数

**类型支持工具函数**
- `IsZero<T>()` (117-139行)：检查类型是否为零，支持float、BFloat16、Half、complex等
- `TypeName<T>()` (142-199行)：返回类型的字符串表示

**TunableOp继承类** (201-321行)
- `GemmTunableOp`：基础可调优GEMM，可选ROCm后端
- `GemmAndBiasTunableOp`：带偏置的可调优GEMM
- `GemmStridedBatchedTunableOp`：批量可调优GEMM
- `ScaledGemmTunableOp`：缩放可调优GEMM，支持多种dtype组合

**主要功能总结：**
- 提供多个BLAS操作的默认实现作为基准
- 通过TunableOp框架注册多个后端实现（rocblas、hipblaslt等）
- 支持浮点8等新数据类型
- 通过`Signature()`生成操作标识用于性能缓存
