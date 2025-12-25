这个文件定义了一个特化的 CUTLASS 模板类，用于处理**去量化整数矩阵乘法操作**（Dequantize Integer GEMM）。

## 核心功能

- **模板特化**：针对 `OpMultiplyAddDequantizeInterleavedBToA` 操作的 `DefaultMmaTensorOp` 类特化
- **数据类型转换**：处理低精度整数类型（int4/int8）到 FP16 的自动转换与计算
- **动态加载指令形状**：根据输入数据类型大小自动计算 K 维度（int8→K=16，int4→K=32）
- **两层指令策略**：
  - `ComputeInstructionShape`：使用原始指令形状进行 FP16 计算
  - `LoadInstructionShape`：适应窄数据类型的内存加载
- **Warp 级 GEMM 操作**：通过 `MmaTensorOpComputeBWithF16` 实现 warp 级别的矩阵乘法
- **Layout 支持**：通过模板参数灵活支持行优先/列优先的矩阵布局
- **分区控制**：支持沿 K 维度分区以优化并行计算
