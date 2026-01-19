这个文件实现了 PyTorch functorch 库中用于**工厂函数（factory functions）的批处理规则（batch rules）**。

## 核心功能

### 1. 模板元编程框架

文件定义了两个核心模板结构，用于自动生成批处理规则：

**`NewBlahBatchRuleHelper`** (lines 33-49)
- 处理接受 `IntArrayRef shape` 参数的工厂函数
- 自动在 shape 前插入 batch 维度
- 例如：`tensor.new_zeros([3, 4])` 在批处理时变成 `tensor.new_zeros([B, 3, 4])`

**`NewBlahBatchRuleHelperSymInt`** (lines 12-30)
- 处理接受 `SymIntArrayRef shape` 参数的工厂函数（支持符号整数）
- 逻辑与上面类似，但使用 `sym_size()` 和 `c10::SymInt`

### 2. 宏定义

```cpp
NEW_BLAH_BATCH_RULE(fn)          // line 54
NEW_BLAH_BATCH_RULE_SYMINT(fn)   // line 60
```

通过函数指针和类型萃取自动实例化模板，简化批处理规则注册。

### 3. 特殊批处理规则

**`_new_zeros_with_same_feature_meta_batch_rule`** (lines 66-105)
- 处理基于另一个张量的元数据创建新张量的情况
- 处理三种情况：
  - Case 1: base 无 batch 维度，tangent 有
  - Case 2: base 有 batch 维度，tangent 无
  - Case 3: 两者都有 batch 维度
- 关键是正确处理 batch 维度的位置

**`linspace_logspace_batch_rule_helper`** (lines 107-143)
- 处理 `linspace` 和 `logspace` 的批处理
- 支持 start/end 为标量或张量的组合
- 当 start 或 end 有 batch 维度时，生成批量的线性/对数空间序列
- 实现了 6 个变体函数（lines 145-220）

**`_has_same_storage_numel_batch_rule`** (lines 222-224)
- 简单返回 `true`，在批处理上下文中假设存储大小相同

### 4. 算子注册

`TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m)` (lines 226-246)

注册了以下工厂函数的批处理支持：
- `ones_like`, `zeros_like`, `empty_like`, `randn_like`, `rand_like`, `full_like`
- `new_empty`, `new_zeros`, `new_ones`, `new_full`
- `linspace` (3 个重载), `logspace` (3 个重载)
- `_new_zeros_with_same_feature_meta`
- `_has_same_storage_numel`

## 设计模式

- **模板元编程**：通过类型萃取自动推导函数签名
- **CRTP 风格**：模板参数包含函数指针本身
- **宏简化**：隐藏复杂的模板实例化细节
- **统一接口**：所有批处理规则返回 `std::tuple<Tensor, std::optional<int64_t>>`

---

**忽略的内容：**
- ROCm 相关：无
- Backward 相关：无（此文件仅处理前向工厂函数）
