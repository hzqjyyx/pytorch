# FunctionalTensorWrapper 核心功能分析

## 主要目的

FunctionalTensorWrapper 是 PyTorch 函数化（Functionalization）机制的核心实现，用于从程序中移除别名（aliasing）和变异（mutation）操作。这对不支持别名的后端（如 XLA、Vulkan）以及需要纯函数式语义的场景（如 Functorch）至关重要。

## 核心机制

### 1. 别名移除（Alias Removal）

**问题**：某些后端无法正确实现 `view()` 等会创建别名的操作。

**解决方案**：
- 将所有 tensor 包装在 `FunctionalTensorWrapper` 中
- 使用共享的 `FunctionalStorageImpl` 存储对象来跟踪别名关系
- 当 `b = a.view(...)` 时，a 和 b 共享同一个 storage，通过它们连接到同一个 Alias 对象

**延迟同步机制**：
- 对任何视图的变异操作会被排队到共享的 Alias 对象上
- 当访问某个 tensor 时，检查其 `generation_` 是否与 storage 的 generation 匹配
- 如果过期（`!is_up_to_date()`），则：
  1. 应用所有待处理的更新到 base tensor（`apply_updates()`）
  2. 从更新后的 base 重新应用视图操作链（`regenerate_from_base()`）

### 2. 变异移除（Mutation Removal）

**转换原理**：
```python
# 原始代码
a.add_(1)

# 转换后
tmp = a.add(1)  # 原地操作变为非原地操作
a.replace_(tmp)  # 通过 replace_ 交换底层 tensor
```

**关键方法 `replace_()`**：
- 将 wrapper 内部的 `value_` tensor 替换为新 tensor
- 同步元数据（sizes, strides, storage_offset）
- 处理 dtype/layout 变化（通过 `_to_copy`）
- 标记变异状态（`mark_mutation()`）
- 追踪是否在 no_grad/inference_mode 下发生

### 3. ViewMeta 栈管理

每个 `FunctionalTensorWrapper` 维护一个 `view_metas_` 向量，记录从 base tensor 到当前 tensor 的完整视图操作链：

```cpp
std::vector<at::functionalization::ViewMeta> view_metas_;
```

**ViewMeta 包含**：
- `forward_fn`: 前向视图函数（如何从 base 生成此视图）
- `reverse_fn`: 反向函数（如何将变异传播回 base）
- `out_index`: 对于多输出视图操作，标识输出索引
- `has_symbolic_inputs`: 是否使用了符号整数

**应用流程**：
```cpp
Tensor apply_view_metas(const Tensor& base) {
  auto t = base;
  for (auto& view_meta : view_metas_) {
    t = view_meta.forward_fn(t, view_meta.out_index);
  }
  return t;
}
```

### 4. 原地视图操作（Inplace View Ops）

对于 `transpose_()` 等既是变异又是视图的操作：
```cpp
void mutate_view_meta(const ViewMeta& meta) {
  view_metas_.push_back(meta);
  has_metadata_mutation_ = true;
  maybe_mark_symbolic(meta);
  
  at::AutoDispatchSkipFunctionalize guard;
  value_ = meta.forward_fn(value_, meta.out_index);
}
```

- 将新 ViewMeta 追加到现有栈
- 立即应用视图操作到底层 `value_`
- 标记为元数据变异

### 5. Storage 管理特殊情况

**`set_()` 操作**：
```cpp
void set__impl(const FunctionalTensorWrapper* other) {
  value_ = other->value_;
  generation_ = other->generation_;
  view_metas_ = other->view_metas_;
  
  functional_storage_impl()->freeze();  // 冻结旧 storage
  storage_ = other->storage_;           // 切换到新 storage
  was_storage_changed_ = true;
}
```

**`resize_()` 操作**：
- 仅支持从 0 字节调整大小或调整到 0 字节
- 禁止对有别名的视图 tensor 增大 storage
- 增大时替换为新的 `FunctionalStorageImpl`（`maybe_replace_storage()`）

**`storage.resize_()` 操作**：
```cpp
void storage_resize_(const c10::SymInt& new_size) {
  // 仅标记，不实际执行操作
  functional_storage_impl()->mark_inductor_storage_resize(new_size);
}
```

## 核心数据结构

### FunctionalTensorWrapper 成员变量

```cpp
Tensor value_;                      // 底层实际 tensor（不含函数化）
int64_t level_;                     // Functorch 层级
bool has_metadata_mutation_;        // 是否发生元数据变异
bool is_multi_output_view_;         // 是否为多输出视图
bool was_storage_changed_;          // 是否调用过 set_()
bool is_symbolic_;                  // 是否使用符号整数
size_t generation_;                 // 当前 generation 计数
std::vector<ViewMeta> view_metas_;  // 视图操作栈
```

### FunctionalStorageImpl

- 虚拟 storage（无实际数据，类似 Meta tensor）
- 维护 Alias 对象引用
- 跟踪全局 `generation_` 计数器
- 队列化待处理的更新操作

## 工具函数 API

### 包装/解包
- `to_functional_tensor()`: 将普通 tensor 包装为 functional tensor
- `from_functional_tensor()`: 提取底层 tensor
- 支持单个 tensor、optional、vector、List 等多种类型

### 同步与更新
- `sync()`: 强制同步 tensor 与其 alias
- `commit_update()`: 提交当前变异到 alias 队列
- `replace_()`: 替换底层 tensor

### 状态查询
- `isFunctionalTensor()`: 检查是否为 functional tensor
- `isBaseTensor()`: 检查是否为 base（非视图）tensor
- `are_all_mutations_hidden_from_autograd()`: 所有变异是否对 autograd 隐藏

### 视图创建
```cpp
Tensor create_functional_tensor_with_view_meta(
    const Tensor& view_to_wrap,
    const Tensor& base,
    functionalization::ViewMeta meta,
    int64_t out_idx = 0
);
```

## 特殊优化与限制

### 自定义元数据方法
重写所有 `_custom()` 方法，直接委托给底层 `value_` tensor：
- `sizes_custom()`, `strides_custom()`, `sym_sizes_custom()` 等
- 确保符号整数（symbolic int）正确传播
- 保证自定义 TensorImpl 的元数据逻辑被调用

### Wrapped Numbers 处理
```cpp
if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    return tensor;  // 直接返回，不包装
}
```
标量 tensor 跳过函数化，因为它们绕过 dispatcher。

### Resize 限制
```cpp
TORCH_CHECK(view_metas_.empty(), 
    "Attempted to resize a view tensor to a larger size. "
    "This is not allowed in the functionalization pass");
```

无法将视图 tensor 的 storage 扩大到更大尺寸，因为会破坏"base 始终包含所有数据"的假设。

### XLA 数据传播
提供专门的 `propagate_xla_data()` 函数处理 XLA 后端的特殊需求（通过 `_propagate_xla_data` 操作）。

## 其他相关点

**ROCm 相关**：
- `CUDASparse.h`, `CublasHandlePool.cpp` 等 CUDA/ROCm 库集成
- 可调优 GEMM 操作（`tunable/TunableGemm.h`）
- HIP/ROCm BLAS 库支持

**Backward 相关**：
- 变异计数器追踪（`mark_mutation_hidden_from_autograd()`）
- No-grad/inference mode 检测
- Version counter 管理（浅拷贝时传播）
