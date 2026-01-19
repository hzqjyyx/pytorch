这个文件实现了 PyTorch functorch 库中卷积操作的批处理规则（batch rules），用于支持 vmap（向量化映射）功能。

## 核心功能

### 1. 前向卷积批处理规则 (`convolution_batch_rule`)

处理输入、权重、偏置三者中任意组合存在批次维度的情况：

**只有输入批处理 (lhs_bdim && !rhs_bdim)**
- 将批次维度合并到 batch 维度
- 执行标准卷积
- 将结果的 batch 维度拆分出批次维度

**只有权重批处理 (!lhs_bdim && rhs_bdim)**
- `groups == 1`: 将批次维度合并到输出通道维度
- `groups > 1`: 
  - 普通卷积：将权重从 `B(GO)IHW` 重塑为 `(GBO)IHW`，输出从 `N(GBO)HW` 重塑为 `BN(GO)HW`
  - 转置卷积：类似处理，但维度顺序不同

**输入和权重都批处理 (lhs_bdim && rhs_bdim)**
- 将两个批次维度都合并到通道维度
- 通过增加 groups 数量来保持计算正确性
- `groups *= batch_size`

**偏置处理**
- 如果权重或偏置有批次维度，先不使用偏置进行卷积
- 卷积完成后单独添加批处理的偏置（broadcast 到空间维度）

### 2. 辅助函数

**`_convolution_decomp`**
- 将 `_convolution` 分解为标准的 `convolution` 调用
- 忽略 benchmark、deterministic、cudnn_enabled 等参数

**`make_dummy`**
- 创建形状正确但内容为空的占位符张量
- 用于 backward 时需要提供形状信息但不需要实际值的场景

### 3. 注册机制

```cpp
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT(convolution, convolution_batch_rule);
  m.impl("_convolution", _convolution_decomp);
  m.impl("convolution_backward", convolution_backward_plumbing);
}
```

将批处理规则注册到 FuncTorchBatched 调度键下。

## 设计思路

通过维度重塑（reshape）和 groups 参数调整，将批处理的卷积转换为等价的非批处理卷积：
- 利用 PyTorch 卷积的 groups 参数实现批次间的独立计算
- 通过 `reshape_dim_into` 和 `reshape_dim_outof` 在批次维度和通道维度间转换

---

**其他内容（简要）：**
- `compute_grad_bias`: 计算偏置梯度（对除通道外的所有维度求和）
- `convolution_backward_input_batch_rule`: 输入梯度的批处理规则
- `convolution_backward_weight_batch_rule`: 权重梯度的批处理规则  
- `convolution_backward_plumbing`: 反向传播的调度入口，处理三个输出的批处理
