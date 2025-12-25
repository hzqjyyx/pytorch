# LegacyBatchedTensorImpl 核心功能

## 整体目的

实现 PyTorch 的 `vmap`（向量化映射）功能的底层张量表示。通过在现有张量上添加"批次维度"来支持自动向量化，而不需要显式地编写批处理代码。

## 核心概念

### BatchDim 结构
- 表示一个"私有"批次维度，是一个 `(level, dim)` 元组
- `level`: 标识该维度是在哪个嵌套 vmap 层级中创建的（支持最多 64 层嵌套）
- `dim`: 指向底层物理张量中被 vmap 处理的维度索引

### BatchedTensorImpl 类
继承自 `TensorImpl`，封装：
- `value_`: 底层实际存储数据的张量
- `bdims_`: 批次维度列表（按 level 递增排序）

批次维度对用户不可见，例如：
```cpp
// 底层张量: ones(2, 3, 5, 7)
// BatchDims: [(lvl=1, dim=0), (lvl=2, dim=1)]
// 用户看到的形状: (5, 7)  // 维度 0 和 1 被隐藏
```

## 主要实现

### 构造函数 (line 9-35)
- 设置 `DispatchKey::Batched` 调度键
- 禁止直接访问存储（`set_storage_access_should_throw()`）
- 计算公开维度：`public_dims = value_.dim() - bdims_.size()`
- 根据实际维度映射设置 `sizes_and_strides_`

### actualDim 方法 (line 37-69)
**核心映射逻辑**：将公开维度索引转换为底层张量的实际维度索引

算法思路：
1. 创建批次维度的位图（bitset）：1 表示批次维度，0 表示普通维度
2. 找到第 `dim` 个为 0 的位置（即第 dim 个非批次维度）

示例（line 44-48）：
```
假设 dim = 3, bitset = 10010011000...
问题：第 3 个（从 0 开始）零在哪里？
答案：索引 5
```

### makeBatched 函数 (line 116-128)
从普通张量创建批次张量：
- 检查张量维度不超过 `kVmapMaxTensorDims` (64)
- 检查批次层级不超过 `kVmapNumLevels` (64)
- 构造 `BatchedTensorImpl` 实例

### addBatchDim 函数 (line 130-141)
为张量添加新的批次维度：
- 如果输入不是批次张量：创建新的批次张量
- 如果已经是批次张量：复制现有 `bdims`，添加新的批次维度，重新创建

### inplaceIsVmapCompatible 函数 (line 143-156)
检查就地操作的 vmap 兼容性：
- 如果 `other` 不是批次张量：兼容
- 如果 `self` 不是批次但 `other` 是批次：不兼容
- 如果两者都是批次：检查 `self` 的层级集合是否包含 `other` 的所有层级

## 限制与约束

### 不支持的操作（抛出断言）
- `set_size()` (line 96-98)
- `set_stride()` (line 99-101)
- `set_storage_offset()` (line 102-104)
- 直接存储访问（line 106-110）

### 内存格式限制 (line 87-92)
`is_contiguous_custom` 仅支持 `MemoryFormat::Contiguous`，其他格式会报错

### 不变量检查 (line 71-77)
`checkInvariants()` 确保 `bdims_` 按 `level` 严格递增排序

## 常量定义

- `kVmapMaxTensorDims = 64`: 最大张量维度数
- `kVmapNumLevels = 64`: 最大嵌套 vmap 层数  
- `kBatchDimsStackSize = 5`: 栈上存储的批次维度数量（小向量优化）

## 辅助函数

- `isBatchedTensor()`: 检查张量是否为批次张量（检查 `DispatchKey::Batched`）
- `unsafeGetBatchedImpl()`: 强制类型转换（不检查）
- `maybeGetBatchedImpl()`: 安全获取实现（先检查类型）
- `createBatchDimBitset()`: 创建批次维度位图
- `createVmapLevelsBitset()`: 创建层级位图

---

**简略说明（忽略内容）：**
- ROCm 相关：无
- Backward 相关：无
