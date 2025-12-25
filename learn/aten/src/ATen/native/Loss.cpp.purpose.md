# Loss.cpp 核心功能分析

这个文件实现了 PyTorch 中各类损失函数的前向计算逻辑。

## 整体架构

- **通用归约辅助函数** (64-71行): `apply_loss_reduction()` 根据 reduction 参数决定是求均值、求和还是不归约
- **Meta函数** (74-101行): 定义 `smooth_l1_loss` 和 `mse_loss` 的元数据，处理输出张量的形状
- **Dispatch机制** (105-110行): 为不同设备（CPU/CUDA）注册具体实现的分发器

## 主要损失函数实现

### 1. Smooth L1 Loss (112-126行)
- 通过 `smooth_l1_stub` 分发到设备特定实现
- 支持 None/Mean/Sum 三种归约方式
- 参数 `beta` 控制平滑程度

### 2. MSE Loss (128-142行)
- 均方误差损失
- 通过 `mse_stub` 分发实现
- 结构与 Smooth L1 类似

### 3. Cosine Embedding Loss (144-180行)
```
计算: 1 - cos(θ) 当 target=1
     max(0, cos(θ) - margin) 当 target=-1
```
- 验证输入维度匹配（0D配1D输入，1D配2D输入）
- 计算余弦相似度时添加 EPSILON (1e-12) 防止除零
- 使用 `where` 选择性应用正负样本损失

### 4. Hinge Embedding Loss (182-197行)
```
计算: x 当 target=1
     max(0, margin - x) 当 target≠1
```
- 特殊处理前向AD中的复合张量，避免原地操作破坏梯度

### 5. Triplet Margin Loss (199-223行)
- 用于度量学习，确保 anchor 与 positive 距离小于与 negative 距离
- 支持 distance swap 技巧（来自论文优化）
- 使用 `pairwise_distance` 计算 L-p 范数距离

### 6. Margin Ranking Loss (225-236行)
```
计算: max(0, -target * (input1 - input2) + margin)
```
- target 为 1 时希望 input1 > input2，为 -1 时相反
- 同样处理前向AD的复合张量情况

### 7. KL Divergence Loss (238-251行)
- 支持两种模式：
  - `log_target=True`: `exp(target) * (target - input)`
  - `log_target=False`: `xlogy(target, target) - target * input`
- 不支持复数和整数输入

### 8. Binary Cross Entropy (253-344行)
**前向计算** (253-300行):
```
L = -w * (y * ln(x) + (1-y) * ln(1-x))
```
- CPU 版本使用 `cpu_kernel` 逐元素计算
- 验证输入输出范围 [0,1]
- 使用 `log1p` 和 `log` 并截断到 -100 防止数值不稳定
- 可选权重加权

**反向传播** (302-344行):
```
d(L)/d(x) = w * (x - y) / (x - x²)
```
- 分母添加 EPSILON 防止除零
- Mean 归约时额外除以元素数量

### 9. Binary Cross Entropy with Logits (346-361行)
- 数值稳定版本，输入为 logits 而非概率
- 支持 `pos_weight` 参数处理类别不平衡
- 内部使用 `log_sigmoid` 避免显式 sigmoid + log

### 10. Poisson NLL Loss (363-378行)
```
log_input=True:  exp(input) - target * input
log_input=False: input - target * log(input + eps)
```
- `full=True` 时添加 Stirling 近似项
- 用于泊松分布的负对数似然

### 11. Soft Margin Loss (380-417行)
```
L = log(1 + exp(-target * input))
```
- 前向使用原地操作链优化: `neg → mul → exp → log1p`
- 反向梯度: `-norm * target * z / (1 + z)` 其中 `z = exp(-target * input)`

### 12. Huber Loss (439-474行)
```
|x-y| < delta:  0.5 * (x-y)²
|x-y| >= delta: delta * (|x-y| - 0.5*delta)
```
- 对异常值鲁棒的损失函数
- 要求 delta > 0
- 通过 `huber_stub` 分发实现

### 13. L1 Loss (494-496行)
```
L = |input - target|
```
- 最简单实现，直接调用 abs 和 reduction

## 关键技术细节

1. **TensorIterator 模式**: 多数损失使用 `TensorIterator::borrowing_binary_op` 高效处理广播和类型提升

2. **前向AD兼容性**: `hinge_embedding_loss` 和 `margin_ranking_loss` 中检查 `_fw_grad(0).defined()` 和 `isTensorSubclassLike`，避免原地操作破坏复合张量的切线

3. **数值稳定性**:
   - BCE 中 log 截断到 -100
   - Cosine loss 添加 EPSILON 到平方和
   - MSE backward 梯度分母保护

4. **内存优化**: 使用 `LEGACY_CONTIGUOUS_MEMORY_FORMAT`、原地操作链、borrowing iterator 减少分配

---

**ROCm 相关**: 文件中无特殊 ROCm 处理

**Backward 相关**:
- `binary_cross_entropy_backward`: BCE 反向传播 (302-344行)
- `soft_margin_loss_backward`: Soft margin 梯度 (380-394行)
- `smooth_l1_loss_backward`: Smooth L1 梯度 (419-437行)
- `huber_loss_backward`: Huber 梯度 (459-474行)
- `mse_loss_backward`: MSE 梯度 (476-492行)
- 各函数通过对应的 `*_backward_stub` 分发到设备实现
