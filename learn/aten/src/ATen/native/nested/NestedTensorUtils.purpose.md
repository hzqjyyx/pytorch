# NestedTensorUtils 核心功能分析

## 主要功能

这两个文件提供了 PyTorch 嵌套张量（NestedTensor）的核心工具函数和实用程序。嵌套张量是一种特殊的张量结构，允许存储不同形状的张量列表，特别适用于变长序列数据（如 NLP 中的批次处理）。

## 核心概念

### 1. 嵌套张量的内部表示

嵌套张量通过以下元数据描述：
- **buffer**: 1D 连续张量，存储所有底层张量的扁平化数据
- **nested_sizes**: 2D 张量，记录每个子张量的形状 (num_tensors × ndim)
- **nested_strides**: 2D 张量，记录每个子张量的步长
- **storage_offsets**: 1D 张量，记录每个子张量在 buffer 中的起始位置

### 2. 构造函数 (NestedTensorUtils.h)

**wrap_buffer()** - 从 buffer 和元数据构造嵌套张量：
```cpp
// 基础版本：仅需 buffer 和 sizes
wrap_buffer(const Tensor& buffer, const Tensor& nested_sizes)

// 完整版本：包含 strides 和 offsets
wrap_buffer(buffer, nested_sizes, nested_strides, storage_offsets)
```

**create_nested_view_tensor()** (NestedTensorUtils.h:83) - 创建视图张量：
- 共享底层存储
- 复制 autograd keys
- 要求在 AutogradFunctionality 禁用的上下文中调用
- 用于实现零拷贝的切片、分块等操作

### 3. 形状信息查询

**NestedTensor_get_sizes()** (NestedTensorUtils.h:109) - 获取所有子张量的 sizes：
- 返回 `vector<IntArrayRef>`，每个元素指向 nested_sizes 的一行
- 零拷贝实现，直接返回指针视图

**NestedTensor_get_max_size()** (NestedTensorUtils.cpp:56) - 计算最大形状：
- 遍历所有子张量，按维度取最大值
- 用于确定填充后的统一形状

**get_consistent_last_dim_of_nested_tensor()** (NestedTensorUtils.cpp:61) - 检查最后一维是否一致：
- 利用 `NestedTensorImpl::opt_size(-1)` 快速检查
- 如果不一致会抛出错误，显示实际的维度值

### 4. 张量分割操作

**chunk_nested_tensor()** (NestedTensorUtils.cpp:70) - 将嵌套张量按最后一维均匀分块：

实现逻辑：
1. 验证条件：
   - 只支持最后一维分块 (dim == ndim - 1)
   - 最后一维必须能被 chunks 整除
   - 要求连续存储
2. 为每个分块创建新的视图元数据：
   ```cpp
   for each split_idx:
     new_sizes[i][dim] = split_size  // 更新大小
     new_offsets[i] = offsets[i] + split_idx * split_size  // 调整偏移
   ```
3. 返回 chunks 个视图张量，共享底层 buffer

**split_with_sizes_nested()** (NestedTensorUtils.cpp:114) - 按指定大小分割：
- 类似 chunk，但支持不等长分割
- 验证 split_sizes 总和等于原始维度大小
- 累积 start_val 计算每个分片的偏移

### 5. 访问器辅助函数 (NestedTensorUtils.h:183-214)

**get_size_for_index()** - 统一接口获取第 i 个元素的 size：
- 嵌套张量：返回 nested_sizes 的第 i 行
- 普通张量：返回 `sizes().slice(1)`（跳过 batch 维度）

**get_stride_for_index()** / **get_offset_for_index()** - 类似逻辑获取 stride 和 offset

这些函数用于编写同时支持嵌套张量和普通张量的通用代码。

### 6. 通用映射框架 (NestedTensorUtils.h:218-447)

**NestedNode<T>** - 树形结构表示嵌套数据：
- `_is_leaf=true`: 存储单个 payload（普通张量）
- `_is_leaf=false`: 存储 children 向量（嵌套张量的子张量列表）

**map()** 函数 - 递归应用函数到嵌套结构：
```cpp
template <class F, class... B>
NestedNode<ReturnType> map(F&& fn, const NestedNode<B>&... nested_node)
```

工作机制：
1. 检查所有输入节点是否都是叶子节点
2. 如果是：直接对 payload 调用 `fn`
3. 如果不是：
   - 广播叶子节点到所有子节点
   - 递归处理每个子节点组合
   - 收集结果到新的 NestedNode

**wrap_tensor_node()** (NestedTensorUtils.h:341) - 将 TensorNode 转换回嵌套张量：

优化路径：
- **快速路径**（所有子张量满足：CPU + 连续 + 相同 dtype）：
  - 分配单个 buffer
  - 使用 `parallel_for` 和 `memcpy` 并行复制数据
  - 直接构造 nested_sizes
  
- **慢速路径**：
  - 逐个 flatten + cat
  - 使用 stack 构建 sizes

**map_nested_tensor()** (NestedTensorUtils.h:440) - 便捷包装器：
```cpp
Tensor result = map_nested_tensor(
    [](Tensor a, Tensor b) { return a + b; },
    nested_tensor_a,
    nested_tensor_b
);
```

## 设计模式

1. **零拷贝视图**：split/chunk 操作通过调整元数据创建视图，不复制数据
2. **类型擦除**：NestedNode 提供统一接口处理嵌套/非嵌套数据
3. **延迟验证**：check_numel_equals_buffer_size 在必要时验证一致性
4. **快慢路径分离**：wrap_tensor_node 针对常见情况优化

## Backward 相关

- **value_selecting_reduction_backward_nested_symint()** (NestedTensorUtils.cpp:170): 仅占位实现，实际逻辑在 Python 端
