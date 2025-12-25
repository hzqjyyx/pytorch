# TensorIndexing 核心功能分析

这两个文件实现了 PyTorch 的 C++ 张量索引系统，提供类似 Python NumPy 风格的张量索引操作。

## TensorIndexing.h - 核心类型和接口

### 1. TensorIndex 类型系统

定义了统一的索引类型 `TensorIndex`，支持 6 种索引方式：

```
- None: 增加维度 (对应 Python 的 None)
- Ellipsis: 省略号展开 (对应 Python 的 ... 或 Ellipsis)
- SymInt: 整数索引选择单个元素
- Boolean: 布尔值索引
- Slice: 切片操作 (start:stop:step)
- Tensor: 张量索引（高级索引）
```

### 2. Slice 结构

封装切片参数，支持符号整数 (SymInt)：
- 默认值处理：step=1, start=0/INDEX_MAX, stop=INDEX_MAX/INDEX_MIN（根据 step 正负）
- 边界检查：step 不能为 0

### 3. 核心操作函数

**get_item** - 读取索引位置的数据
- 单索引快速路径优化（整数、切片、None、Ellipsis、布尔）
- 多维索引通过 `applySlicing` 处理
- 高级索引（张量索引）通过 `dispatch_index` 处理

**set_item** - 设置索引位置的数据
- 类似 get_item 的分支处理
- 通过 `copy_to` 执行实际数据复制
- 支持标量赋值和张量赋值

### 4. 维度处理核心逻辑 (handleDimInMultiDimIndexing)

处理单个维度的索引操作：
- **整数索引**: 调用 `applySelect` → `tensor.select_symint(dim, index)`
- **切片**: 调用 `applySlice` → `tensor.slice_symint(dim, start, stop, step)`
  - 优化：当 start=0, stop=size, step=1 时直接返回原张量
- **Ellipsis**: 跳过多个维度 (`dim += original_tensor.dim() - specified_dims`)
- **None**: 插入新维度 (`unsqueeze`)
- **布尔**: 转换为索引张量（true→[0], false→[]）
- **张量**: 记录到 `outIndices` 用于高级索引

### 5. 辅助功能

- `count_specified_dimensions`: 计算实际索引维度数（排除 Ellipsis 和 None）
- `slicePrefix1sSize`: 移除左侧的 size=1 维度（NumPy 兼容性）
- `boolToIndexingTensor`: 布尔值转索引张量
- `scalarToTensor`: 标量转张量（设备感知优化）

## TensorIndexing.cpp - 实现和接口绑定

### 1. 流输出操作符

提供调试支持：
- `operator<<(Slice)`: 输出 "start:stop:step" 格式
- `operator<<(TensorIndex)`: 根据类型输出不同格式
- `operator<<(vector<TensorIndex>)`: 输出索引列表 "(index1, index2, ...)"

### 2. Tensor 类成员函数实现

**Tensor::index**
```cpp
Tensor::index(ArrayRef<TensorIndex> indices)
  → OptionalDeviceGuard(device)
  → at::indexing::get_item(*this, indices)
```

**Tensor::index_put_**
```cpp
Tensor::index_put_(ArrayRef<TensorIndex> indices, Tensor/Scalar value)
  → OptionalDeviceGuard(device)
  → at::indexing::set_item(*this, indices, value)
```

提供 `initializer_list` 重载以支持 `tensor.index({0, 1})` 语法。

### 3. 标量赋值特殊处理

`set_item` 的标量版本：
- 根据设备类型转换标量为张量
- QInt 类型特殊处理：先转 CPU float 再赋值
- CUDA: 先在 CPU 创建再传输
- 其他设备: 直接在目标设备创建

## 关键设计考虑

1. **性能优化**
   - `disable_slice_optimization`: 控制是否跳过冗余切片（如 `x[0:size]`）
   - 单索引快速路径避免向量分配
   - 静态标量张量优化（CPU 设备）

2. **Python 兼容性**
   - 函数镜像 `torch/csrc/autograd/python_variable_indexing.cpp`
   - 保持独立实现避免 Python 调用开销（热路径）
   - NumPy 语义兼容（如 slicePrefix1sSize）

3. **符号整数支持**
   - 所有索引参数使用 `SymInt` 支持动态形状
   - 边界检查使用符号比较 (`sym_gt`, `sym_eq`)

4. **嵌套张量处理**
   - 通过 `optional<SymIntArrayRef>` 传递 size（嵌套张量为 nullopt）
   - NOTE [nested tensor size for indexing] 标记相关代码

---

**ROCm 相关**: 无特殊 ROCm 代码

**Backward 相关**: 
- 使用 `AutoDispatchBelowADInplaceOrView` guard 在标量赋值时跳过自动微分
- 索引操作本身通过调度系统支持自动求导
