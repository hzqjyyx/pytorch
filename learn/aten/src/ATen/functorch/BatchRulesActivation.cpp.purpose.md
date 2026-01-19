这个文件实现了 PyTorch 的 functorch 模块中激活函数的批处理规则。

**主要功能：**

- **GLU 函数的批处理规则** (`glu_batch_rule`)：处理 Gated Linear Unit 激活函数在批维度下的计算，通过将批维度移到前面，调用标准的 `at::glu` 实现

- **GLU 反向传播的批处理规则** (`glu_backward_batch_rule`)：处理 GLU 反向传播时的梯度计算，确保梯度和输入张量的批维度对齐

- **VMAP 支持注册**：通过 `TORCH_LIBRARY_IMPL` 宏将这些批处理规则注册到 functorch 的 FuncTorchBatched 分发键中，使得 vmap（向量化映射）能够正确处理这些操作

**核心设计：**

- 利用 `moveBatchDimToFront` 将批维度统一移到张量前面
- 使用 `rankWithoutBatchDim` 计算不含批维度的秩，正确调整操作维度
- 通过 `ensure_has_bdim` 确保梯度和输入的批维度一致性
