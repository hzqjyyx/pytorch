这个文件实现了 PyTorch functorch 库中对各种归约操作（reduction operations）的批处理规则（batch rules），用于支持 vmap（向量化映射）功能。

## 核心机制

**boxed_reduction_batch_rule 模板函数** (107-236行)
这是文件的核心，是一个通用的批处理规则处理器：

- 接受三个模板参数：
  - `dim_arg_pos`: dim 参数的位置
  - `keepdim_case`: keepdim 的处理方式（False/True/Variable）
  - `maybe_keepdim_arg_pos`: keepdim 参数的位置（如果存在）

- 主要处理流程：
  1. 从栈中提取参数，解包带批次维度的张量
  2. 将批次维度移到最前面
  3. 处理 dim 参数（支持 int、IntList、None 三种类型）
  4. 将逻辑维度转换为物理维度
  5. 处理标量张量的特殊情况（dim=0 或 dim=-1）
  6. 调用原始操作
  7. 将结果重新包装为带批次维度的张量

**标量张量特殊处理** (174-208行)
当逻辑维度为 0 且 dim 为 0/-1 时：
- 通过 unsqueeze 添加一个维度
- 执行操作后根据 keepdim 决定是否 squeeze
- 这样处理 `vmap(lambda x: x.sum(0))(torch.tensor([10.]))` 这类边界情况

## 辅助函数

**分解函数（decomposition functions）**
将无维度参数的归约操作分解为有维度参数的版本：
- `sum_decomp`: 对所有维度求和
- `mean_decomp`: 对所有维度求均值
- `prod_decomp`: 先 flatten 再对第 0 维求积
- `max_decomp/min_decomp`: 先 flatten 再求最大/最小值
- `norm_scalar_decomp`: 对所有维度求范数
- `median_decomp/nanmedian_decomp`: 先 flatten 再求中位数
- `all_decomp/any_decomp`: 先 flatten 再判断全部/任意为真

**searchsorted_batch_rule** (318-405行)
处理 searchsorted 操作的批处理规则，需要区分两种情况：
- `buckets_logical_rank > 1`: 多维边界数组
- `buckets_logical_rank == 1`: 一维边界数组
根据哪些输入有批次维度，采用不同的扩展和重塑策略

**expand_bdims** (244-258行)
确保两个张量的批次维度对齐，将没有批次维度的张量扩展到有批次维度的形状

## 宏定义

提供三个便捷宏用于注册批处理规则：

- `REDUCTION_BOXED_ARGS`: 通用宏，需要指定所有参数
- `REDUCTION_WITH_KEEPDIM_ARG`: 假设 dim 在位置 1，keepdim 在位置 2
- `REDUCTION_NO_KEEPDIM_ARG`: 假设 dim 在位置 1，操作总是保持秩不变

## 操作注册 (450-511行)

在 `TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m)` 中注册了大量归约操作：
- FFT 操作：`_fft_r2c`, `_fft_c2r`, `_fft_c2c`
- 统计操作：`amax`, `amin`, `aminmax`, `mean`, `sum`, `prod`, `std`, `var`
- 逻辑操作：`all`, `any`
- 排序操作：`argmax`, `argmin`, `sort`, `topk`, `kthvalue`, `mode`
- 累积操作：`cumsum`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`
- 范数操作：`norm`, `linalg_vector_norm`, `logsumexp`
- 搜索操作：`searchsorted`, `bucketize`
- 其他：`count_nonzero`, `dist`, `rot90`

---

**简要列出的其他内容：**
- ROCm 相关：无（文件中未涉及）
- Backward 相关：
  - `_softmax_backward_batch_rule` (260-288行)
  - `_log_softmax_backward_batch_rule` (290-316行)
  - 两者都通过扩展张量并调用原始反向传播函数实现
