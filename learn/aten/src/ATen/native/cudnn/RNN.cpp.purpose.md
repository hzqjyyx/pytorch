# aten/src/ATen/native/cudnn/RNN.cpp 主要功能分析

## 核心职责

这个文件实现了 PyTorch 与 cuDNN RNN 库的接口层，将 PyTorch 的 RNN 操作（LSTM、GRU、RNN_RELU、RNN_TANH）转换为 cuDNN API 调用。

## 关键组件

### 1. 参数描述符封装

**DropoutDescriptorParams** (124-144行)
- 管理 dropout 配置（训练模式、dropout率、随机状态）
- 根据是否启用 dropout 创建对应的 cuDNN descriptor

**RNNDescriptorParams** (148-282行)
- 存储 RNN 核心参数：hidden_size、proj_size（LSTM projection）、num_layers、mode（LSTM/GRU等）
- 处理双向 RNN 配置
- 支持 cuDNN v7 和 v8 API 的条件编译（USE_CUDNN_RNN_V8_API）
- 生成 cuDNN RNNDescriptor

**TensorDescriptorListParams** (423-486行)
- 处理三种输入格式：
  - **Packed sequence**: 变长序列打包格式（batch_sizes 非空）
  - **Unpacked (seq_first)**: [seq_len, batch, input_size]
  - **Unpacked (batch_first)**: [batch, seq_len, input_size]
- 为 cuDNN 创建对应的 tensor descriptors

### 2. 权重管理

**get_parameters()** (696-870行)
- 将 cuDNN 的扁平化权重缓冲区切分为独立的 weight/bias 张量视图
- 处理 LSTM 的特殊布局：`(reset, forget, cell, output)` gates
- 处理 GRU 的特殊布局：`(reset, input, new)` gates
- 支持 LSTM with projection 的额外权重矩阵（add_projection_weights）

**_cudnn_rnn_flatten_weight()** (1383-1408行)
- 将 PyTorch 的分层权重列表（每层的 weight_ih, weight_hh, bias_ih, bias_hh）合并为 cuDNN 要求的单一连续缓冲区
- 通过 `copy_weights_to_flat_buf_views()` 实现双向转换

### 3. 算法选择启发式

**get_algo()** (1194-1224行)
根据硬件和参数特性选择最优算法：

```
CUDNN_RNN_ALGO_STANDARD          // 通用算法
CUDNN_RNN_ALGO_PERSIST_STATIC    // 持久化算法（FP16优化）
CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H  // 小 hidden size 优化
```

**use_persist_device_heuristics()** (1119-1165行)
- **Volta (SM7.0)**: 特定 batch size 和 seq_length 组合
- **Ampere (SM8.x)**: batch size 8的倍数且≤128，排除 GRU
- **Turing (SM7.5)**: 禁用持久化算法
- **特殊排除**: A40 (sm_86) 因 cuDNN 8.0.5 的边缘情况被排除

### 4. 前向传播实现

**_cudnn_rnn()** (1453-1738行)

**输入处理流程**:
```
1. 参数验证
   - check_attributes() 检查设备、dtype 一致性
   - LSTM 才允许 cx（cell state）
   
2. Transpose 处理（如果 batch_first=true）
   input.transpose(0, 1)  // [batch, seq, input] → [seq, batch, input]

3. 权重缓冲区准备
   - 如果未提供 weight_buf: 警告 "weights not contiguous"
   - 分配扁平缓冲区并复制权重
   
4. Workspace 分配
   - 训练模式: workspace + reserve（保存中间结果用于反向传播）
   - 推理模式: 仅 workspace

5. cuDNN API 调用
   v7: cudnnRNNForwardTraining / cudnnRNNForwardInference
   v8: cudnnRNNForward (统一接口，通过 CUDNN_FWD_MODE_* 区分)
```

**返回**: (output, hy, cy, reserve, weight_buf)

### 5. 数据类型提升

**promote_rnn_math_type()** (1226-1231行)
- FP16 输入 → FP32 内部计算（精度保护）
- 其他类型保持不变

### 6. 尺寸计算辅助函数

```cpp
_input_size()   // 根据 packed/unpacked 格式返回输入形状
_hidden_size()  // 考虑 projection: 使用 proj_size 而非 hidden_size
_cell_size()    // LSTM cell state: 始终用 hidden_size
_output_size()  // 双向 RNN: hidden_size * num_directions
```

## 重要特性

### cuDNN v8 API 差异
- **v7**: 使用 FilterDescriptor 描述权重，传递 TensorDescriptor 数组
- **v8**: 使用 RNNDataDescriptor，权重通过 size+pointer 传递
- 通过 `USE_CUDNN_RNN_V8_API` 宏实现条件编译

### LSTM Projection 支持
- proj_size ≠ 0 时启用
- 添加额外的 `w_hr` 权重（linear_id=8）
- 强制使用 STANDARD 算法（persistent 不支持）
- hidden state 输出维度变为 proj_size

### Packed Sequence 处理
- cuDNN v8 需要将 batch_sizes 转换为 seqLengthArray
- 示例: batch_sizes=[3,2,1,1] → seqLengthArray=[4,2,1]
- Persistent 算法对 packed input 支持有限（通常返回 CUDNN_STATUS_NOT_SUPPORTED）

---

## ROCm 和 Backward 相关（概要）

- **ROCm 支持**: 通过相同的代码路径，cuDNN 宏定义兼容 MIOpen
- **Backward 实现**:
  - `_cudnn_rnn_backward_input()`: 计算 dx, dhx, dcx（数据梯度）
  - `_cudnn_rnn_backward_weight()`: 计算权重梯度（需在 backward_input 后调用）
  - 使用 reserve 缓冲区恢复前向传播的中间状态
- **权重视图管理**: `_viewParams()` / `_copyParams()` 处理分层权重与扁平缓冲区的双向映射
- **无 cuDNN 编译**: 提供 stub 实现抛出错误信息
