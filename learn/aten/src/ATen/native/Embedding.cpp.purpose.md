这个文件实现了 PyTorch 中的 Embedding 操作的核心功能。

**主要函数：**

- `embedding_symint()` - 根据索引从权重矩阵中查找嵌入向量。支持动态形状（SymInt），当索引是1D时直接使用 `index_select`，多维索引时先扁平化再索引最后重塑
- `embedding_backward_symint()` - 嵌入层反向传播的分发函数，根据 `sparse` 参数选择稀疏或密集梯度计算
- `embedding_sparse_backward()` - 稀疏梯度反向传播，构造 COO 格式稀疏张量作为梯度，支持过滤 padding_idx
- `embedding_dense_backward_cpu()` - CPU 上的密集梯度反向传播，使用并行循环累加梯度到权重矩阵，支持 `scale_grad_by_freq` 按频率缩放
- `embedding_renorm_cpu_()` - 对嵌入权重进行范数归一化，限制最大范数以稳定训练

**核心特性：**

- 支持符号整数（SymInt）处理动态形状
- 支持 padding_idx 参数，允许某个索引对应的嵌入向量为零且不参与梯度更新
- 密集反向传播使用 TensorIterator 和并行化处理大规模数据
- 支持按出现频率缩放梯度（`scale_grad_by_freq`）来处理不平衡数据集
