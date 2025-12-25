## 主要数据结构

### 1. CellParams 系列 - 权重参数封装

**CellParams**（174-220行）：标准 RNN cell 的权重容器
- `w_ih`: input-to-hidden 权重矩阵
- `w_hh`: hidden-to-hidden 权重矩阵  
- `b_ih`, `b_hh`: 偏置向量（可选）
- `w_hr`: 投影矩阵（仅 LSTM with projections）
- 提供 `matmul_ih()`, `linear_ih()` 等统一接口

**QuantizedCellParams**（228-300行）：量化版本参数
- 额外存储 `packed_ih/hh`, `col_offsets`, `scale`, `zero_point`
- 使用 `fbgemm_linear_int8_weight_fp32_activation` 执行量化矩阵乘法

### 2. Cell 抽象 - 单步计算

**SimpleCell**（706-717行）：基础 RNN cell
```cpp
Tensor operator()(input, hidden, params) {
    return nonlinearity(params.linear_hh(hidden) + params.linear_ih(input))
}
```
通过模板参数 `nonlinearity` 实现 Tanh-RNN 和 ReLU-RNN

**LSTMCell**（721-756行）：LSTM 核心逻辑
- **GPU 路径**（731-741行）：调用 `_thnn_fused_lstm_cell` kernel，支持投影层
- **CPU 路径**（743-753行）：手动实现门控机制
  ```
  gates = linear_hh(hx) + linear_ih(input)
  ingate, forgetgate, cellgate, outgate = gates.chunk(4)
  cy = forgetgate * cx + ingate * cellgate
  hy = outgate * tanh(cy)
  ```

**GRUCell**（759-788行）：GRU 实现
- GPU 路径使用 `_thnn_fused_gru_cell`
- CPU 路径实现 reset/input/new gate 逻辑

### 3. Layer 抽象 - 序列扫描

**FullLayer**（820-861行）：标准序列输入处理
- `operator()(std::vector<Tensor>)`: 逐步迭代调用 cell，累积输出
- `operator()(Tensor)`: 展开序列维度后调用上述方法
- **CPU 优化**（846-852行）：预计算 `linear_ih(inputs)` 避免重复计算

**PackedLayer**（928-976行）：PackedSequence 支持
- 处理变长序列的 `batch_sizes` 索引
- 动态切片隐藏状态以匹配当前 batch size
- 前向遍历序列，batch size 递减时保存隐藏状态

**ReversedPackedLayer**（979-1030行）：反向 PackedSequence
- 从最小 batch 开始反向遍历
- batch size 递增时从 `input_hidden` 扩展隐藏状态

**Bidirectional Layer 变体**（864-917行，1033-1058行）：
- 组合前向和反向 layer
- 拼接双向输出 `cat({fw, rev}, -1)`

### 4. 多层堆叠

**apply_layer_stack**（1076-1099行）：
```cpp
for l in 0..num_layers:
    layer_output = layer(layer_input, hiddens[l], weights[l])
    layer_input = layer_output.outputs
    if l < num_layers-1 and train:
        layer_input = dropout(layer_input, dropout_p)
```

## 核心函数流程

### RNN/GRU 主函数（宏 ONE_HIDDEN_RNN，1183-1313行）

**两个重载版本**：
1. **标准输入**（batch_first 可选）
2. **PackedSequence 输入**

**执行路径选择**：
```
if cudnn_is_acceptable(input):
    → cudnn_stub (cuDNN 加速)
else if use_miopen(input, dropout_p):
    → miopen_stub (ROCm 加速)  
else:
    → CPU fallback: _rnn_impl_with_concat
```

**CPU fallback 流程**：
```cpp
1. gather_params() 组织参数为 CellParams 数组
2. batch_first → transpose(0,1)
3. _rnn_impl_with_concat():
   - 调用 apply_layer_stack
   - stack 所有层的最终隐藏状态
4. batch_first → transpose 回去
```

### LSTM 主函数（1428-1525行）

类似路径选择，增加了：
- **MPS 后端支持**（1440-1445行）：Apple Silicon GPU
- **MKL-DNN 支持**（1461-1475行）：Intel CPU 优化
- **投影检测**（1448行）：`hx[0].size(2) != hx[1].size(2)`，MiOPEN/MKL-DNN 不支持时降级

**隐藏状态转置**（1139-1146行）：
```cpp
// 输入: hx[num_layers, batch, hidden], cx[num_layers, batch, cell]
// 转为: hiddens[num_layers] 每个元素是 tuple(hx, cx)
```

## 关键设计模式

### 1. 模板多态组合
```cpp
_rnn_impl<CellType, LayerT, BidirLayerT>(...)
// 实例化为:
// - GRUCell + FullLayer + FullBidirectionalLayer
// - LSTMCell + PackedLayer + PackedBidirectionalLayer
```

### 2. 双路径优化
- **GPU**: 调用融合 kernel (`_thnn_fused_lstm_cell`)
- **CPU**: 预计算 input 变换减少冗余计算

### 3. PackedSequence 处理
通过 `batch_sizes` 数组动态管理不同长度序列的批处理，避免 padding

---

## ROCm 和 Backward 相关（简要）

- **ROCm**: `use_miopen()` 检测和 `*_miopen_stub` 分发，功能等价 cuDNN
- **Backward**: 
  - `_thnn_fused_lstm_cell_backward` (1170-1176行)
  - `_thnn_differentiable_*_cell_backward` (28-29行引用)
  - `sigmoid_backward`, `tanh_backward` (57, 60行引用) 用于手动梯度计算
