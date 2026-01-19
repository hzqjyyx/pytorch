## ZeroTensorFallback.cpp 核心功能

这个文件实现了 PyTorch 中 ZeroTensor 的 fallback 机制，用于处理对零张量（ZeroTensor）的操作。

### 设计理念

ZeroTensor 是不可变的（immutable）。当需要修改时，必须先通过 `.clone()` 将其物化（materialize）为普通张量。

### 主要逻辑流程

**1. 参数分析阶段（lines 19-34）**
- 遍历算子的所有参数，检查 alias_info
- 判断操作是否为写操作（`isWrite()`）
- 确保不存在混合的可变/不可变别名参数

**2. 只读操作处理（lines 36-42）**
- 如果是非写操作（view 操作），直接 redispatch
- 假设 view 操作会自动传播 ZeroTensor dispatch key

**3. 参数物化阶段（lines 44-80）**
- 遍历栈上的参数
- 对于 Tensor 类型：
  - 如果是 ZeroTensor 且为可变参数 → 报错
  - 如果是 ZeroTensor 且为只读 → 物化为 `zeros({}).expand(sizes)`
- 对于 TensorList 类型：
  - 对列表中每个 ZeroTensor 执行相同检查和物化

**4. 重新分发（line 82）**
- 将处理后的参数重新分发到下一个 dispatch key

### 注册机制

**通用 fallback（lines 86-88）**
```cpp
TORCH_LIBRARY_IMPL(_, ZeroTensor, m)
```
为所有命名空间注册默认 fallback 处理函数

**特殊操作豁免（lines 90-106）**
```cpp
TORCH_LIBRARY_IMPL(aten, ZeroTensor, m)
```
某些操作使用 `makeFallthrough()`，直接跳过 ZeroTensor 处理：
- `zeros_like`, `mul.Scalar`, `add.Scalar` - 结果仍为零
- `copy_`, `clone` - 需要直接操作
- `dot`, `vdot` - 点积运算
- 宏展开的 view 函数和工具函数

### 关键错误检查

**不可变性保护（lines 61-62, 73-74）**
```cpp
TORCH_CHECK(!mut_arg, "ZeroTensors are immutable...")
```
防止对 ZeroTensor 执行 in-place 操作

**一致性检查（lines 24-29）**
确保算子的别名参数要么全部可变，要么全部不可变

---

**ROCm/Backward 相关**：
- 无 ROCm 特定代码
- 无反向传播相关实现（line 71 TODO 提到应检查 `requires_grad=False`）
