# FunctionalStorageImpl 核心功能解析

## 核心目的

FunctionalStorageImpl 是 PyTorch 函数化（Functionalization）机制的核心组件，用于在没有实际数据指针的情况下追踪和处理张量的视图关系及变更操作。它是一个特殊的 StorageImpl 子类，类似 meta storage，不包含实际数据。

## 主要机制

### 1. 视图元数据追踪（ViewMeta）

ViewMeta 记录了视图操作的前向和反向函数：

- **forward_fn**: 如何在基础张量上重放视图操作
- **reverse_fn**: 如何从变更后的视图恢复到基础张量
- **关键属性**:
  - `out_index`: 多输出视图的索引
  - `is_multi_output`: 是否为多输出视图
  - `is_as_strided`: 是否为 as_strided 操作
  - `has_symbolic_inputs`: 是否有符号化输入

### 2. 变更队列机制（Update）

```cpp
struct Update {
    const at::Tensor new_val;           // 变更后的值
    const std::vector<ViewMeta> view_metas;  // 视图链
}
```

**工作流程**（以代码注释中的例子为例）：

```cpp
base = ...
a = base.view1()
b = a.view2()
c = b.view3()
c.add_(3)  // 变更操作
```

调用 `add_update()` 时会存储：
- `new_val = c`（变更后的 c）
- `view_metas = [view1_meta, view2_meta, view3_meta]`

### 3. 更新应用算法（apply_update）

这是整个机制的核心，采用**前向构建 + 反向应用**的策略：

**前向阶段**（构建中间视图）：
```cpp
tmp_values = [base]
tmp_values.append(view1(base))      // tmp_values = [base, a]
tmp_values.append(view2(a))         // tmp_values = [base, a, b]
// 注意：不包括最后一个视图 c
```

**反向阶段**（应用逆变换）：
```cpp
t = new_val  // t = 变更后的 c
t = view3_inverse(b, t, 0)  // 用 b 的形状信息恢复
t = view2_inverse(a, t, 0)  // 用 a 的形状信息恢复
t = view1_inverse(base, t, 0)  // 最终得到变更后的 base
base_ = t
```

**为什么需要 tmp_values**：某些视图操作（如 select/slice/diagonal/squeeze/as_strided）的逆操作需要原始张量的尺寸信息才能正确恢复。

### 4. 存储大小计算（get_nbytes）

处理多种特殊情况：
- **稀疏张量**: 返回 0（因为没有传统的连续存储）
- **符号化尺寸**: 
  - Python 模式（Proxy Tensor）：使用 `storage().sym_nbytes()`
  - 其他：通过 `computeStorageNbytes` 计算
- **XLA 张量**: 使用计算值而非存储对象的 nbytes
- **常规张量**: 基于 sizes/strides/itemsize/offset 计算

### 5. 代数计数器（Mutation Counters）

追踪三类变更：

```cpp
mutation_counter_                                    // 总变更数
mutation_counter_during_no_grad_or_inference_mode_   // no_grad 下的变更
mutation_counter_hidden_from_autograd_               // 对 autograd 完全隐藏的变更
```

**用途**：判断变更是否可以安全地保留在编译图中
- 普通变更：可能改变 autograd 元数据（如 `.grad_fn`），需要在图外重放
- no_grad 变更：不改变元数据但会增加版本计数器，需要 `mark_dirty()`
- hidden 变更（如 Triton kernel）：完全隐藏，可保留在图中

### 6. 存储调整追踪

```cpp
mark_inductor_storage_resize(new_size)
```

记录 Inductor 编译器引起的存储调整，保存原始和当前大小以判断是否为 nop。

## 关键设计决策

### 防止引用循环

`base_` 始终指向函数化层**下方**的张量：

```
FunctionalTensorWrapper(a)
  └─ storage: FunctionalStorageImpl
       └─ base_: a.value_ (未包装的值)
              ↓
       FunctionalTensorWrapper(b)
         └─ storage: 同一个 FunctionalStorageImpl
              └─ base_: 同样指向 a.value_
```

### as_strided 限制

在编译模式下，如果视图链长度 > 1 且包含 `as_strided()`，会抛出错误（除 XLA 外）。原因：as_strided 是非组合操作，无法正确函数化。

### 冻结机制

```cpp
freeze()  // 设置后不可逆
```

冻结后的存储不允许任何变更，用于保证某些优化的正确性。

---

**ROCm 相关**: 无直接相关内容

**Backward 相关**: 
- autograd 元数据追踪（grad_fn 变更检测）
- 版本计数器管理（mark_dirty 决策）
- no_grad/inference_mode 下的变更特殊处理
