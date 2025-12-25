## `fused_adam_utils.cuh` 文件功能解析

这是一个 **PyTorch CUDA 优化器工具文件**，用于实现融合的 Adam 优化器算法。主要功能如下：

### 核心组件

1. **ADAM_MODE 枚举** (第 10 行)
   - `ORIGINAL`: 标准 Adam 算法
   - `ADAMW`: AdamW 算法（含权重衰减改进）

2. **adam_math 函数** (第 26-90 行)
   - **参数索引定义**：
     - `kParamIdx=0`: 模型参数
     - `kGradIdx=1`: 梯度
     - `kExpAvgIdx=2`: 一阶动量（exponential moving average）
     - `kExpAvgSqIdx=3`: 二阶动量
     - `kMaxExpAvgSqIdx=4`: AMSGrad 的最大二阶动量（可选）

   - **核心计算逻辑**：
     ```
     1. 加载参数、梯度和动量状态
     2. 处理梯度缩放（用于混合精度训练）
     3. 应用权重衰减（ADAM vs ADAMW 不同处理）
     4. 更新一阶动量：exp_avg = β₁·exp_avg + (1-β₁)·grad
     5. 更新二阶动量：exp_avg_sq = β₂·exp_avg_sq + (1-β₂)·grad²
     6. 计算有偏差修正的学习率
     7. 如果启用 AMSGrad，取二阶动量的最大值
     8. 更新参数：param -= lr·exp_avg / (√exp_avg_sq + eps)
     9. 存储更新后的状态
     ```

3. **FusedAdamMathFunctor 结构体** (第 104-197 行)
   - **融合优化器的 CUDA 核函数**
   - 支持偏差修正和 AMSGrad
   - 支持梯度缩放（混合精度训练的 GradScaler）
   - 支持无穷值检测（防止 NaN 训练）

### 关键特性

| 特性 | 说明 |
|------|------|
| **融合计算** | 在单个 CUDA 核中完成整个参数更新，减少内存访问 |
| **矢量化** | 使用 `kILP=4` (Instruction Level Parallelism) 进行向量化 |
| **混合精度支持** | 支持 FP16 训练（梯度缩放和无穷值检测） |
| **多种精度** | 通过模板支持不同的计算精度类型 |
| **AMSGrad** | 可选的动量修正（使用最大二阶动量） |

### 应用场景

这个文件被 PyTorch 的 `torch.optim.Adam` 和 `torch.optim.AdamW` 在 CUDA 设备上使用，特别是在：
- 大规模模型训练
- 混合精度（AMP）训练
- 需要高性能的优化器场景
