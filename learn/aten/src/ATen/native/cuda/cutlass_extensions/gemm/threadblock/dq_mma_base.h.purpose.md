## dq_mma_base.h 文件分析

这是一个 CUTLASS 库中用于量化 GEMM (General Matrix Multiply) 操作的基础类模板。

### 核心结构

**DqMmaBase 类** - 双缓冲的线程块范围 GEMM 内核模板，包含以下关键组件：

**模板参数：**
- `Shape_`: GEMM 问题的大小
- `Policy_`: 调优策略配置
- `ElementScale_`: 缩放因子类型
- `Stages`: 流水线级数

**SharedStorage 嵌套结构：** 存储线程块级别的共享内存
- `operand_A`: A 矩阵操作数缓冲
- `operand_B`: B 矩阵操作数缓冲
- `operand_scale`: 量化缩放因子缓冲 (大小为 Shape::kN)

**关键类型定义：**
- `WarpGemm`: 单个 warp 计算的 GEMM 形状
- `WarpCount`: CTA 中 warp 的网格布局
- `kWarpGemmIterations`: 每个 warp 的 GEMM 迭代次数

### 功能要点

- **量化支持**: 集成量化缩放因子存储与计算
- **Warp 级迭代**: 通过 `kNumKIterationsPerWarpBLoad` 和 `kWarpGemmIterationsForB` 管理 K 维迭代分解
- **SFINAE 模板重载**: `run_warp_mma()` 函数支持不同 WarpMma 配置（处理标准与变换片段）
- **内存布局**: 支持自定义内存对齐和填充（SmemPaddingA/B）
- **迭代器管理**: 维护 A/B 操作数的 warp 级共享内存迭代器

### 关键特性总结

- 双缓冲流水线架构，用于高效的 GEMM 计算
- 量化感知的矩阵乘法基础设施
- 灵活的 warp 级工作分解与调度
- CUTLASS 框架的线程块范围抽象层
