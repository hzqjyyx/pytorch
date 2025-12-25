这个文件实现了 LSTM 和 GRU 两种 RNN 单元的 CUDA 前向计算核心功能：

## LSTM 前向计算 (lstm_cell_forward)

**核心计算流程** (aten/src/ATen/native/cuda/RNN.cu:100-181):

1. **输入门 (input gate)**: `ig = sigmoid(iig + hig + b1i + b2i)`
2. **遗忘门 (forget gate)**: `fg = sigmoid(ifg + hfg + b1f + b2f)` 
3. **候选值 (cell gate)**: `cg = tanh(icg + hcg + b1c + b2c)`
4. **输出门 (output gate)**: `og = sigmoid(iog + hog + b1o + b2o)`
5. **新细胞状态**: `cy = fg * cx + ig * cg`
6. **新隐藏状态**: `hy = og * tanh(cy)`

**数据组织**:
- 输入和隐藏状态的门控信号按 4 倍隐藏维度组织 (input/forget/cell/output)
- 每个线程处理一个隐藏单元，通过 `linearIndex` 映射到 4 组门控值
- 中间结果 (ig, fg, cg, og) 保存到 workspace 供反向传播使用

## GRU 前向计算 (gru_cell_forward)

**核心计算流程** (aten/src/ATen/native/cuda/RNN.cu:252-316):

1. **重置门 (reset gate)**: `rg = sigmoid(ir + hr + b1r + b2r)`
2. **更新门 (update gate)**: `ig = sigmoid(ii + hi + b1i + b2i)`
3. **候选隐藏状态**: `ng = tanh(in + b1n + rg * (hn + b2n))`
4. **新隐藏状态**: `hy = ng + ig * (hx - ng)` 即 `(1-ig)*ng + ig*hx`

**数据组织**:
- 门控信号按 3 倍隐藏维度组织 (reset/update/new)
- workspace 按 5 倍隐藏维度组织，保存 rg, ig, ng, hx, (hn+b2n)

## 实现特性

**性能优化**:
- `getLaunchConfig()` 根据元素数量动态计算 grid/block 配置
- `allContiguous()` 检测张量连续性，连续时用 `indexing_kind=1` 简化索引计算
- `collapseDims()` 折叠维度减少索引开销
- 32/64位索引动态选择 (`canUse32BitIndexMath`)

**数值精度处理**:
- `H2F/F2H` 宏处理 half/float 类型转换
- 使用 `accscalar_t` (累加标量类型) 进行中间计算，避免精度损失
- 支持 Float/Half/BFloat16 类型 (通过 `AT_DISPATCH_FLOATING_TYPES_AND2`)

**Bias 处理**:
- 可选的 input_bias 和 hidden_bias (通过 `has_bias` 标志)
- undefined 的 bias 用零值替代
- Half 类型下显式转换零值 `F2H(0.0)`

---

**Backward 相关**: 反向传播实现了梯度计算，LSTM 计算 grad_gates/grad_cx，GRU 额外分离 input/hidden gates 梯度

**ROCm 相关**: `USE_ROCM` 宏启用 AMD GPU 支持，与 CUDA 共享大部分代码路径
