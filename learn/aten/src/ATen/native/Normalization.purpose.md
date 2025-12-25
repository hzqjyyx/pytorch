# Normalization.cpp/h 主要功能

这两个文件实现了PyTorch中的**归一化操作**，主要包含Batch Normalization和相关的归一化功能。

## 核心功能模块

### 1. Batch Normalization (批归一化)

**主入口函数**: `batch_norm()` (line 678-712)
- 支持训练和推理两种模式
- 参数包括输入张量、权重、偏置、运行均值/方差、动量、epsilon
- 内部调用 `_batch_norm_impl_index()` 选择后端实现

**后端选择机制**: `_select_batch_norm_backend()` (line 497-542)
根据以下条件选择最优后端：
- **CuDNN**: CUDA设备 + 满足特定条件(维度、数据类型、batch size限制等)
- **MIOpen**: AMD GPU + 满足特定条件
- **Native**: CPU或不满足加速库条件时的原生实现

### 2. Native CPU实现

**前向传播核心函数**: `batch_norm_cpu()` (line 814-861)

分为两个阶段：

**阶段1 - 统计量更新** (训练模式):
- `batch_norm_cpu_update_stats_template()` (line 200-306)
  - 计算当前batch的均值和方差
  - 使用指数移动平均更新running_mean和running_var
  - 公式: `running_mean = momentum × batch_mean + (1-momentum) × running_mean`
  - 无偏方差估计: `var_sum / (n-1)`

**阶段2 - 输入变换**:
- `batch_norm_cpu_transform_input_template()` (line 135-197)
  - 归一化公式: `output = ((input - mean) × invstd) × weight + bias`
  - 支持连续和非连续内存布局
  - 快速路径: 所有张量连续时调用优化的stub

**优化策略**:
- **连续内存路径** (line 141-155): 使用 `batch_norm_cpu_stub` 向量化实现
- **非连续路径** (line 157-196): 使用TensorIterator逐元素计算
- **混合精度**: 支持Half输入+Float参数 (line 176, 211-218)

### 3. 辅助功能

**Instance Normalization**: `instance_norm()` (line 714-750)
- 通过reshape将实例归一化转换为批归一化
- 将 `(B, C, H, W)` reshape为 `(1, B×C, H, W)`
- 对每个channel单独计算统计量

**Renorm操作**: `renorm_out()` (line 951-984)
- 限制张量沿某个维度的范数不超过maxnorm
- 计算向量范数 → 生成缩放因子 → 应用缩放

### 4. 关键数据结构和模板

**VarTransform模板** (line 103-119):
- `InvStd`: 将方差转换为 `1/sqrt(var + eps)` (用于前向)
- `Var`: 保持原始方差(用于统计更新)

**内存格式处理** (line 121-132):
- `is_contiguous()`: 检查Contiguous/ChannelsLast/ChannelsLast3d
- `suggest_memory_format_contig()`: 推断最佳内存格式

### 5. Dispatch机制

定义在Normalization.h中的stub (line 8-9):
```cpp
DECLARE_DISPATCH(renorm_scale_factor_fn, renorm_scale_factor_stub)
```

在Normalization.cpp中定义 (line 84-87):
```cpp
DEFINE_DISPATCH(batch_norm_cpu_stub);
DEFINE_DISPATCH(batch_norm_cpu_collect_stats_stub);
DEFINE_DISPATCH(batch_norm_cpu_backward_stub);
```

这些stub在运行时根据CPU架构选择最优实现(AVX2/AVX512等)。

## 数学原理

**训练时**:
1. 计算batch统计: `mean = E[x]`, `var = E[(x-mean)²]`
2. 归一化: `x_norm = (x - mean) / sqrt(var + eps)`
3. 缩放平移: `y = γ × x_norm + β`
4. 更新运行统计: `running_mean = m × mean + (1-m) × running_mean`

**推理时**:
- 直接使用running_mean/running_var进行归一化
- 无需计算batch统计量

## 简要提及的内容

**Backward相关**:
- `batch_norm_backward_cpu()` (line 929-948): 计算梯度
- `_batch_norm_impl_index_backward()` (line 631-675): 根据前向backend调用对应backward
- 梯度计算公式在line 440-467 (训练模式) 和 line 469-483 (推理模式)

**ROCm/MIOpen相关**:
- MIOpen后端选择条件 (line 523-539)
- `at::miopen_batch_norm` 调用 (line 615, 672)
- 维度限制 `MIOPEN_DIM_MAX = 5` (line 65)
