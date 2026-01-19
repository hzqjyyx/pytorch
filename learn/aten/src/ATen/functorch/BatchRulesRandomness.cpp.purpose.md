这个文件实现了 PyTorch functorch 中随机操作的批处理规则（batching rules）。与常规批处理规则不同，这些规则注册到 `FuncTorchVmapMode` 而非 `FuncTorchBatched`，因为需要拦截所有随机操作，即使它们不在 BatchedTensor 上。

## 核心机制

**RandomnessType 控制**：
- `Different`：每个批次元素使用不同的随机数（默认行为）
- `Same`：所有批次元素共享相同的随机数
- `Error`：禁止随机操作

## 主要批处理规则模板

**1. `random_batching_rule`** (lines 22-37)
- 用于创建新张量的随机操作（如 `randn`, `rand`）
- `Different` 模式：在 shape 前添加 batch 维度，调用底层函数，返回 BatchedTensor
- `Same` 模式：直接调用底层函数，不添加 batch 维度

**2. `random_inplace_batching_rule`** (lines 39-62)
- 用于就地随机操作（如 `normal_`, `uniform_`）
- 检查：`Different` 模式下不允许对 unbatched tensor 进行就地操作（会看起来像 `Same`）
- `Same` 模式 + batched input：创建临时张量，填充随机数，再 copy 回去

**3. `randperm_batching_rule`** (lines 110-127)
- 特殊处理排列操作
- `Different` 模式：循环生成 batch_size 个独立排列，stack 起来
- `Same` 模式：生成单个排列

**4. `unary_pointwise_random_batch_rule`** (lines 129-153)
- 用于基于输入张量形状的随机操作（如 `bernoulli(tensor)`）
- `Different` + unbatched：expand 输入以添加 batch 维度
- `Same` + unbatched：不添加 batch 维度到输出

**5. `tensor_like_random_batch_rule`** (lines 155-178)
- 用于 `*_like` 操作（如 `randn_like`, `rand_like`）
- `Same` + batched：取第一个切片 `[0]` 作为模板
- `Different` + unbatched：expand 以添加 batch 维度

## 特殊实现

**`multinomial_batching_rule`** (lines 225-262)
- 处理 1D 和 2D 输入
- `Different` 模式：reshape 为 2D，调用 multinomial，再 reshape 回来
- `Same` 模式：仅支持 unbatched 输入

**`bernoulli_inplace_Tensor_batching_rule`** (lines 64-108)
- 处理 `bernoulli_(tensor, p_tensor)` 的复杂情况
- 对齐两个输入的维度（padding 到相同 logical rank）
- 检查 inplace 兼容性

**`native_dropout_batching_rule`** (lines 180-218)
- eval 模式不检查 randomness
- `Same` 模式：手动实现 dropout（调用 `bernoulli_` 生成 mask）
- `Different` 模式：对 unbatched 输入 expand 后调用原生 dropout

## 宏注册系统

使用模板元编程和宏批量注册操作：
- `RANDOM_BATCH_RULE`: 注册创建张量的随机操作
- `RANDOM_INPLACE_BATCH_RULE`: 注册就地随机操作
- `RANDINT_BATCH_RULE`: 处理参数顺序特殊的 randint
- `RANDPERM_BATCH_RULE`: 注册 randperm
- `UNARY_POINTWISE_RANDOM`: 注册基于输入的随机操作

注册到两个 dispatch key：
- `FuncTorchVmapMode` (lines 376-502)：大部分随机操作
- `FuncTorchBatched` (lines 365-374)：`bernoulli_.float`

---

**忽略的内容**：
- `native_dropout_backward_batch_rule` (line 220)：反向传播规则
- ROCm 相关：文件中无 ROCm 特定代码
