这个文件定义了 PyTorch functorch 中对**动态形状操作**的批处理规则。

**核心功能：**

- **禁止动态形状操作的 vmap**：文件中的所有规则都是抛出错误，阻止用户在 vmap 中使用会产生动态输出形状的操作

- **unsupportedDynamicOp 宏**：为 `nonzero`、`where`、`unique_dim` 等操作注册处理器，这些操作的输出形状取决于输入数据内容，无法在 vmap 中批处理

- **特殊错误处理**：
  - `_local_scalar_dense`、`item`：禁止从张量提取标量值
  - `is_nonzero`：禁止数据相关的控制流
  - `allclose`：不支持 vmap

- **注册机制**：通过 `TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m)` 在 FuncTorchBatched 分发键下注册这些不支持的操作

**关键点：**

- 这不是实现功能，而是**明确拒绝**不兼容的操作
- 所有错误消息都指向 GitHub issues，鼓励用户反馈需求
- 适用于 vmap（向量化映射）场景，其中批维度需要统一的输出形状
