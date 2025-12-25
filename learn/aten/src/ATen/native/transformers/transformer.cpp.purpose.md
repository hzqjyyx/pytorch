- **文件目的**: 实现 PyTorch Transformer 编码层的前向传播计算

- **主要函数**:
  - `transformer_encoder_layer_forward()` (73行): 完整的 Transformer 编码层前向计算
  - `ffn()` (44行): 前馈网络 (Feed-Forward Network)
  - `linear_for_ffn()` (24行): 线性层 + 可选激活函数
  - `norm()` (61行): 层归一化 (Layer Normalization)

- **核心计算流程**:
  1. 可选的前置层归一化 (`norm_first`)
  2. 多头自注意力 (`_native_multi_head_attention`)
  3. 残差连接 (`x.add_(src)`)
  4. 后置层归一化 (如果 `norm_first=false`)
  5. 前馈网络 (两层线性变换)
  6. 再次残差连接

- **特殊支持**:
  - 嵌套张量 (Nested Tensor) 优化路径
  - 可选的注意力掩码 (mask)
  - GELU 激活函数支持
  - 空张量处理

- **依赖**: 调用高层 ATen 操作 (`_native_multi_head_attention`, `_addmm_activation`, `addmm`, `layer_norm`)，不含底层 CUDA 实现
