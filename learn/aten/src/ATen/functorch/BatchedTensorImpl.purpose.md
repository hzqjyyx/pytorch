# BatchedTensorImpl 核心功能

## 主要目的
实现 vmap (向量化映射) 功能的底层张量包装器，将批处理维度从用户可见的张量维度中"隐藏"起来。

## 核心概念

**批处理维度的隐藏机制**
- `BatchedTensorImpl` 包装一个底层 `Tensor`，并标记其中一个维度为批处理维度 (batch dimension)
- 该批处理维度对用户不可见，所有公开的维度索引都会自动跳过批处理维度
- 例如：`BatchedTensorImpl(ones(2, 3, 5, 7), bdim=0, level=1)` 
  - 底层张量形状：`(2, 3, 5, 7)`
  - 用户看到的形状：`(3, 5, 7)`
  - 维度 0 被隐藏为批处理维度

## 关键数据成员

- `value_`: 底层实际存储数据的 Tensor
- `bdim_`: 批处理维度的索引（在底层张量中的位置）
- `level_`: vmap 嵌套层级（支持最多 64 层嵌套）

## 核心方法

### actualDim() - 维度映射
```cpp
int64_t actualDim(int64_t dim, bool wrap_dim) const
```
将公开的维度索引转换为底层张量的实际维度索引：
- 如果 `bdim_ <= dim`，返回 `dim + 1`（跳过批处理维度）
- 否则返回 `dim`（批处理维度之前的维度保持不变）

### refreshTensorMetadata()
更新张量的元数据（sizes, strides, storage_offset）：
- **非嵌套张量**：遍历所有公开维度，通过 `actualDim()` 从底层张量获取对应的 size 和 stride
- **嵌套张量**：只支持 `bdim=0` 且 `level=1` 的单层 vmap

### 尺寸和步幅查询
- `size_custom(d)` / `sym_size_custom(d)`: 返回指定维度的大小
- `sizes_custom()` / `sym_sizes_custom()`: 返回所有维度的大小（嵌套张量不支持）
- `strides_custom()` / `sym_strides_custom()`: 返回步幅信息

## 工厂函数

**makeBatched()**
```cpp
Tensor makeBatched(Tensor tensor, int64_t bdim, int64_t level)
```
- 创建 `BatchedTensorImpl` 实例
- 传播特定的 DispatchKey（Negative, Conjugate, XLA, CUDA, CPU）
- 检查嵌套的 batched tensor 的 level 递增约束

**addBatchDim()**
```cpp
Tensor addBatchDim(Tensor tensor, int64_t dim, int64_t level)
```
便捷包装，直接调用 `makeBatched()`

## 辅助函数

- `isBatchedTensor()`: 检查张量是否为 BatchedTensor
- `maybeGetBatchedImpl()`: 安全获取 BatchedTensorImpl 指针
- `unsafeGetBatchedImpl()`: 不安全的类型转换（假设已确认是 BatchedTensor）
- `createBatchDimBitset()`: 创建批处理维度的位集合
- `createVmapLevelsBitset()`: 创建 vmap 层级的位集合

## 限制和约束

### 嵌套张量限制
- 只能在 `dim=0` 上进行 vmap
- 只支持单层 vmap (`level=1`)

### 禁止的操作
以下方法被覆写并抛出错误：
- `set_size()` / `set_stride()`: 不允许直接修改尺寸
- `shallow_copy_and_detach()`: 不允许在 vmap 内访问 `.data`
- `shallow_copy_from()`: 不允许通过 `.data` 直接修改
- 不支持 storage 访问（`set_storage_access_should_throw()`）

### 内存格式
`is_contiguous_custom()` 只支持查询 `MemoryFormat::Contiguous`

## 配置常量

- `kVmapMaxTensorDims = 64`: 最大张量维度数
- `kVmapNumLevels = 64`: 最大 vmap 嵌套层级
- `kBatchDimsStackSize = 5`: 栈上存储的 BatchDims 数量
- `kKeysToPropagateToWrapper`: 需要传播到包装器的 DispatchKey 集合

## Unsafe 方法（内部使用）

- `_unsafe_set_level()`: 修改 level（用于 torchdim 的非词法作用域 BatchedTensor）
- `unsafe_set_bdim()`: 修改 bdim（用于 in-place view 操作如 squeeze_/unsqueeze_，调用后必须调用 `refreshTensorMetadata()`）

---

**简要提及：**
- 文件不涉及 ROCm 相关内容
- 文件不涉及 Backward/梯度计算相关内容
