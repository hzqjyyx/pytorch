这个文件实现了 PyTorch functorch 库中二元操作（binary operations）的批处理规则（batch rules），用于支持 vmap（向量化映射）功能。

## 核心功能

### 1. 二元逐点操作批处理
`_binary_pointwise_batch_rule` (17-26行) 是核心模板函数，处理两个张量的逐点二元操作：
- 接收两个张量及其各自的批次维度信息
- 调用 `_binary_pointwise_helper` 对齐两个张量的批次维度
- 执行实际的二元操作并返回结果

### 2. 随机二元操作批处理
`BinaryRandomPointwiseBatchRuleHelper` (50-86行) 处理涉及随机性的二元操作（如 `normal`）：
- 检查随机性类型（`RandomnessType::Different` 或 `Same`）
- `Different` 模式：确保输出对每个批次元素都不同，需要扩展非批次输入
- `Same` 模式：所有批次元素使用相同随机值，可直接调用原操作

### 3. 原地二元操作批处理
`binary_pointwise_inplace_batch_rule` (94-119行) 处理原地修改操作（如 `add_`, `mul_`）：
- 检查兼容性：如果左操作数无批次维度但右操作数有，则报错（无法原地广播）
- 计算最大逻辑秩并对齐维度
- 执行原地操作

### 4. 比较操作批处理
`comparison_pointwise_batch_rule` (122-142行) 处理比较操作（如 `eq`, `gt`, `lt`）：
- 类似二元逐点操作，但专门用于返回布尔张量的比较操作
- 对齐批次维度和逻辑秩后执行比较

### 5. 特殊操作
- `where_self_batch_rule` (144-160行)：处理三元条件选择 `torch.where(condition, self, other)`
- `masked_select_batch_rule` (177-194行)：处理掩码选择，不支持对 mask 进行 vmap
- `fill__Tensor_batch_rule` (262-279行)：优化的填充操作

## 宏定义系统
文件使用大量宏来批量注册操作：
- `BINARY_POINTWISE_BATCH_RULE`：生成标准二元操作的批处理规则
- `BINARY_RANDOM_POINTWISE_BATCH_RULE`：生成随机二元操作的批处理规则
- `BINARY_SCALAR_2/3`：处理 Tensor×Tensor、Tensor×Scalar、Scalar×Tensor 的组合
- `COMPARISON_POINTWISE`：注册比较操作

## 注册的操作类型
在 `TORCH_LIBRARY_IMPL` 块中注册了大量操作（342-549行）：
- 算术运算：`add`, `sub`, `mul`, `div`, `pow` 等
- 位运算：`bitwise_and`, `bitwise_or`, `bitwise_xor`, 位移操作
- 比较运算：`eq`, `gt`, `ge`, `le`, `lt`, `ne`
- 逻辑运算：`logical_and`, `logical_or`, `logical_xor`
- 数学函数：`atan2`, `hypot`, `fmax`, `fmin`, `gcd`, `lcm` 等
- 特殊函数：`polygamma`, `igamma`, `xlogy` 等

---

**ROCm 相关**: 无

**Backward 相关**:
- `gelu_backward_batch_rule` (162-175行)
- `masked_select_backward_batch_rule` (196-219行)
- `cdist_backward_batch_rule` (221-260行)
- `log_sigmoid_backward_batch_rule` (320-336行)
- 注册多个激活函数的反向传播：`elu_backward`, `sigmoid_backward`, `tanh_backward`, `silu_backward` 等
