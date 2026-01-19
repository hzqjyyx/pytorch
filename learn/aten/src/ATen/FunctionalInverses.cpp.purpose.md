这个文件实现了 PyTorch 的 **functionalization pass** 中的视图操作逆函数（view inverses）。

## 核心目的

Functionalization pass 的目标是消除张量操作中的别名（aliasing），将所有视图操作转换为拷贝操作，同时保持语义正确性。当对视图进行修改时，需要通过"逆函数"将修改传播回原始张量。

## 工作原理

以一个典型场景为例：

```cpp
// 原始代码（有别名）
view1 = input1.view_op(args...)
view1.add_(1)  // 修改 view1，input1 也被修改

// functionalization 后（无别名）
view_copy1 = input1.view_copy_op(args...)
view_copy1.add_(1)  // 修改 view_copy1，input1 不受影响
input1 = view_op_inverse(input1, view_copy1, args...)  // 通过逆函数同步修改
```

## InverseReturnMode

控制逆函数的返回行为：
- **NeverView**: 总是返回拷贝（用于 functionalization）
- **AlwaysView**: 返回视图（用于 autograd）
- **ViewOrScatterInverse**: 根据情况选择

## 主要实现的逆函数类别

### 1. 简单对称操作
- `permute_inverse`: 反转维度置换
- `transpose_int_inverse`: 再次转置恢复
- `t_inverse`: 转置的逆就是转置
- `unsqueeze_inverse`: 通过 squeeze 恢复

### 2. 需要 base 信息的操作
- `squeeze_inverse`: 需要知道原始哪些维度是 1
- `view_inverse`: 需要原始 shape
- `_reshape_alias_inverse`: 恢复原始形状

### 3. 使用 scatter 操作的逆函数
- `select_int_inverse`: 使用 `select_scatter` 将修改写回特定索引
- `slice_Tensor_inverse`: 使用 `slice_scatter` 写回切片
- `diagonal_inverse`: 使用 `diagonal_scatter` 写回对角线
- `split_*_inverse`: 计算偏移量后用 `slice_scatter` 写回

### 4. 特殊处理
- `expand_inverse`: 使用 `sum_to` 聚合扩展维度的修改
- `as_strided_inverse`: 使用 `as_strided_scatter` 处理任意步长视图
- `unfold_inverse`: 调用 `unfold_backward`，检查内部重叠

### 5. 复数/共轭视图
- `view_as_real_inverse` ↔ `view_as_complex`
- `view_as_complex_inverse` ↔ `view_as_real`
- `_conj_inverse`: 共轭的逆是共轭
- `_neg_view_inverse`: 负视图的逆是负视图

### 6. 简单传递
- `detach_inverse`: 直接返回（functionalization 不关心 autograd 元数据）
- `lift_fresh_inverse`: 直接返回
- `alias_inverse`: 根据模式返回 alias 或 alias_copy

---

**不支持的操作（抛出错误）：**
- Sparse tensors 相关：`_indices`, `_values`, `indices`, `values`, `crow_indices`, `col_indices` 等
- Forward AD 相关：`_fw_primal`, `_make_dual`
- Nested tensors：`_nested_view_from_buffer`（部分支持 jagged nested tensors）
