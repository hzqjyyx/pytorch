这个文件实现了用于 Tensor Core 矩阵乘法的**运行时反量化**功能，主要用于混合精度 GEMM 计算中对量化权重的动态缩放。

## 核心功能

### 主模板类 `MmaTensorOpDequantizer`
- 为 warp 级别的矩阵乘法操作提供反量化能力
- 针对不同的 GPU 架构（Volta/Turing/Ampere）和数据类型（FP16/BF16）提供特化实现
- 仅处理 Operand B（通常是权重矩阵）

### 关键操作流程
1. **初始化阶段**（构造函数）：
   - 根据 warp 索引和线程索引计算在共享内存中的缩放因子位置
   - 存储指向缩放因子数组的指针

2. **加载阶段**（`load` 方法）：
   - 从共享内存加载缩放因子到寄存器片段（`FragmentScale`）
   - 每个 MMA 迭代在 N 维度上加载对应的缩放值

3. **反量化阶段**（`dequantize` 方法）：
   - 将量化的操作数片段与缩放因子相乘
   - 输出反量化后的操作数片段用于 Tensor Core 计算

## 架构特化实现

### Ampere (SM80+) + BF16 特化 (lines 87-190)
- 使用 `__nv_bfloat162` 向量化乘法（`__hmul2`）
- 2 元素打包处理提升性能
- 仅在 `__CUDA_ARCH__ >= 800` 时编译

### Turing/Ampere (SM75+) + FP16 特化 (lines 195-283)
- 使用 CUTLASS 的 `multiplies` functor
- 支持 ColumnMajor B 布局
- 通过 `kExpansionFactor` 处理加载/计算指令比例差异

### Volta (SM70) + RowMajor B 特化 (lines 287-370)
- 针对 32x32x4 交错 GEMM
- 每个 MMA tile 处理 8 个元素
- 使用 `AccessType` (8 元素数组) 批量加载缩放因子

### Volta (SM70) + ColumnMajor B 特化 (lines 374-466)
- 同样针对 32x32x4 交错 GEMM
- 线程访问模式不同：`base_col = lane_idx & 0xF8 + lane_idx % 4`
- 每个 tile 迭代内部再分 2 个 MMA 迭代，步长为 4

## 设计要点

- **片段寄存器化**：所有数据通过 `Array` 模板存储在寄存器中，避免内存访问
- **编译期优化**：大量使用 `CUTLASS_PRAGMA_UNROLL` 和 `static_assert` 进行循环展开和类型检查
- **线程协作模式**：每个线程处理特定的缩放因子子集，由 quad ID 和 warp 位置决定
- **零拷贝转换**：使用 `reinterpret_cast` 在不同数据类型视图间切换

---

**ROCm/Backward 相关**：
- 无 ROCm 特定代码
- 无反向传播实现（仅前向推理反量化）
