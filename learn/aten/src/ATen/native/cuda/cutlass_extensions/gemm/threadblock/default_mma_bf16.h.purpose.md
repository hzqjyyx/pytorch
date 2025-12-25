这个文件定义了针对 bfloat16 数据类型的 GEMM (通用矩阵乘法) 操作的线程块级别的默认 MMA (Matrix Multiply-Accumulate) 配置。

## 核心架构

文件通过模板特化 `DefaultMma` 来为不同的输入组合提供优化实现：

### 1. BF16 × BF16 基础实现 (lines 41-110)

针对 bfloat16 激活值和 bfloat16 权重的通用实现：

- **架构兼容性处理**：通过 `arch_has_bf16_mma` 检测计算能力是否 ≥ 80 (Ampere+)
  - Ampere+ 架构：直接使用 bfloat16 进行 MMA
  - 早期架构：转换为 half_t 后再进行 MMA
- **流水线策略**：使用 `MmaPipelined`，适用于 2-stage 流水线
- **迭代器**：使用 `PredicatedTileIterator` 处理边界条件

### 2. BF16 × BF16 Ampere 优化实现 (lines 139-210)

专门针对 SM80 (Ampere) 架构的特化版本：

- **多阶段流水线**：使用 `MmaMultistage` 替代 `MmaPipelined`
- **设计意图**：通过 2-stage 多阶段流水线避免大 tile 时的寄存器溢出
- **技巧**：MmaCore 使用 3-stage 配置触发多阶段组件，但实际 MMA 使用 2-stage
- **迭代器升级**：使用 `PredicatedTileAccessIterator` 支持异步拷贝 (cp.async)

### 3. 混合精度量化支持

**BF16 × INT8 (lines 236-286, 388-441)**
- 激活值：bfloat16
- 权重：uint8_t (8-bit 量化)
- 使用 `DqMma` (Dequantize MMA) 封装，包含反量化逻辑
- Scale 对齐：`kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value`

**BF16 × INT4 (lines 311-361, 470-523)**
- 激活值：bfloat16
- 权重：uint4b_t (4-bit 量化)
- 同样使用 `DqMma` 实现
- 支持多阶段流水线版本 (kStages 参数化)

## 关键设计模式

1. **类型转换层**：早期架构通过 `MmaElementA/B` 条件类型实现 bf16→fp16 转换
2. **委托模式**：量化版本将实现委托给 `DqMma` 模板
3. **分层抽象**：
   - MmaCore：核心计算单元配置
   - Iterator：数据加载策略
   - ThreadblockMma：线程块级别的完整 MMA 实现

## 性能优化点

- **共享内存管理**：`SharedMemoryClear` 参数控制越界访问处理
- **Gather 支持**：`GatherA/B` 参数支持索引数组间接访问
- **对齐优化**：`kAlignmentA/B` 确保内存访问对齐
- **缓存策略**：Ampere 版本使用 `kCacheOpA/B` 控制缓存行为

---

**ROCm 相关**: 无

**Backward 相关**: 无
