这个文件定义了 CUTLASS GEMM（通用矩阵乘法）的配置系统，用于 FasterTransformer 库中的 GPU 加速计算。

**CutlassTileConfig 枚举**：定义了不同的 CTA（Cooperative Thread Array）和 Warp 形状组合
- `Undefined` 和 `ChooseWithHeuristic` 用于动态选择配置
- SiMT 配置：`128x128x8_WarpShape64x64x8`（基础 SIMT 计算模式）
- TensorCore 配置：多种 CTA 和 Warp 组合，针对不同的矩阵规模（M=32/64/128）优化，CTA_N 固定为 128，CTA_K 为 64

**SplitKStyle 枚举**：K 维度分割策略
- `NO_SPLIT_K`：不分割
- `SPLIT_K_SERIAL`：串行分割

**CutlassGemmConfig 结构体**：实际的配置参数
- `tile_config`：使用的瓦片配置
- `split_k_style`：K 维度分割方式
- `split_k_factor`：分割因子
- `stages`：流水线阶段数

**主要功能**：
- 为 CUTLASS 库的 GEMM 操作提供可配置的硬件映射参数
- 支持多种 GPU 计算模式（SiMT 和 TensorCore）
- 用于权重量化（weight-only quantization）时的内核布局配置
