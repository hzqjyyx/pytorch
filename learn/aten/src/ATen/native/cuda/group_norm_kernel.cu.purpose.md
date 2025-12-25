# Group Norm CUDA Kernel 核心功能分析

这个文件实现了 Group Normalization 的 CUDA 前向和反向传播内核。

## 核心计算流程

**Forward Pass 主要步骤：**

1. **计算统计量** (`RowwiseMomentsCUDAKernel` 32-72行)
   - 对每个 group 计算均值和标准差的倒数
   - 使用 Welford 在线算法进行数值稳定的均值/方差计算
   - 通过 warp/block reduce 在线程间聚合结果
   - 输出：`mean[i]` 和 `rstd[i] = 1/sqrt(variance + eps)`

2. **归一化和仿射变换**
   - **1D 情况** (`GroupNorm1dForward` 482-551行)：直接用 TensorIterator 执行 `(x - mean) * rstd * gamma + beta`
   - **多维情况** (554-634行)：
     - 如果没有 gamma/beta：直接归一化 `(x - mean) * rstd`
     - 如果有 gamma/beta：先通过 `ComputeFusedParamsCUDAKernel` (75-97行) 预计算融合参数
       - `a = rstd * gamma`
       - `b = -rstd * gamma * mean + beta`
     - 然后执行 `y = a * x + b`

## 关键数据布局

```
输入张量形状：[N, C, H, W]
- N: batch size
- C: channels (必须能被 G 整除)
- G: groups 数量
- D = C / G: 每个 group 的 channel 数
- HxW: 空间维度
```

每个 group 独立计算均值和方差，即共有 `N * G` 个统计量。

## 性能优化技术

1. **Reduction 优化**
   - 小数据用 warp reduce (单个 warp 内)
   - 大数据用 block reduce (多个 warp 协作，使用 shared memory)
   - 减少同步开销

2. **Kernel Fusion**
   - 融合归一化和仿射变换到单个 kernel
   - 通过预计算 `a` 和 `b` 参数减少重复计算

3. **TensorIterator 框架**
   - 自动处理广播、类型转换、内存合并访问
   - 支持多种输入组合（有/无 gamma、有/无 beta）

4. **数值精度处理**
   - Half/BFloat16 输入时用 Float 累加 (`acc_type<T, true>`)
   - 避免低精度累加导致的数值误差

## 特殊情况处理

- **HxW = 1**：使用专门的 1D 优化路径
- **无 gamma/beta**：跳过仿射变换，减少计算
- **N = 0**：空 batch 直接返回

---

**Backward Pass 简要说明：**
- 反向传播计算三个梯度：`dX`（输入梯度）、`dgamma`、`dbeta`
- 使用类似的 reduction 策略和融合参数技术
- 针对不同 batch size (N ≤ 128 vs N > 128) 选择不同的列归约算法

**ROCm 支持：**
- 代码中未见 ROCm 特定逻辑，主要通过 PyTorch 的抽象层（如 `AT_CUDA_CHECK`）兼容
