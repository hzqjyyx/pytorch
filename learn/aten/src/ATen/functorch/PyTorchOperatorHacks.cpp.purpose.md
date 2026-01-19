这个文件包含了 functorch 对一些有问题的 PyTorch 复合算子的临时修复实现。

## 核心问题

PyTorch 的某些复合算子（composite operators）在 functorch 的转换下无法正常工作，主要原因是：
- 使用了 in-place 操作
- 调用了 `data_ptr()` 等底层接口
- 与 vmap 等转换不兼容

## 主要修复的算子

**1. linear_hack (lines 35-71)**
- 处理线性层 `y = xW^T + b` 的计算
- 针对不同输入维度选择最优实现路径：
  - 2D 输入：使用 `addmm` 融合操作
  - 3D 连续输入：reshape 后使用 `addmm`，再 reshape 回来
  - 其他情况：`matmul` + `add`
- **关键修复**：检测是否存在 vmap 层，如果有则使用 `add()` 而非 `add_()` 避免 in-place 操作（lines 62-68）

**2. binary_cross_entropy_with_logits_hack (lines 82-108)**
- 实现带 logits 的二元交叉熵损失
- 使用数值稳定的计算方式：`max_val = (-input).clamp_min(0)` 避免溢出
- 支持 `pos_weight` 参数用于正样本加权
- 支持 reduction 模式（mean/sum/none）

**3. dropout 系列 hack (lines 124-255)**
- 重新实现了 dropout 及其变体（feature_dropout, alpha_dropout）
- **核心改动**：
  - 使用 `at::empty()` + `bernoulli()` 生成 mask，而非原始实现中可能有问题的方式（lines 177-185）
  - feature_dropout 使用 `make_feature_noise()` 生成特定形状的 noise（保留 batch 和 channel 维度，其他维度为 1）
- alpha_dropout 使用 SELU 激活函数配套的特殊 dropout 变体
- 模板参数 `inplace` 控制是否原地修改

## 注册机制

通过 `TORCH_LIBRARY_IMPL` 将这些 hack 实现注册到 `FuncTorchDynamicLayerFrontMode` dispatch key（lines 257-272），使得 functorch 转换时会调用这些修复版本而非原始实现。

---

**其他内容：**
- `index_select_backward_hack`: 反向传播实现
- `trace_backward_decomp`: 矩阵 trace 的反向传播
- ROCm/XNNPACK 相关的特殊路径处理
