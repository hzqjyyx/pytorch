## TensorGeometry 主要功能

**TensorGeometry** 是一个轻量级的数据结构，用于表示和检查张量的几何属性（形状、步长等），而无需构造完整的张量对象。

### 核心设计目的
- 允许基于子张量的几何信息选择计算核心策略
- 缓存张量的连续性状态，避免重复计算
- 支持符号形状和步长（SymInt），用于动态形状推理

### 主要功能

**_geometry_is_contiguous 函数** (cpp 第7-25行)
- 模板函数，检查给定的 sizes 和 strides 是否表示连续张量
- 从最后一维向前遍历，验证步长是否符合连续内存布局
- 处理空张量（size为0）的特殊情况，返回 true

**geometry_is_contiguous 函数** (cpp 第27-29行)
- 公共接口，包装 _geometry_is_contiguous

**TensorGeometry 结构体** (h 第15-152行)

*构造方式：*
- 从 SymIntArrayRef（符号整数数组）构造：自动计算步长
- 从 TensorBase 对象构造：提取其大小、步长、偏移量
- 完全指定构造：提供 sizes、strides、storage_offset

*几何信息访问：*
- `sym_sizes()`, `sym_strides()`, `sym_storage_offset()`, `sym_numel()`：符号版本
- `sizes()`, `strides()`, `storage_offset()`, `numel()`：整数版本（需非符号张量）
- `dim()`, `size(dim)`, `stride(dim)`：维度相关查询

*几何变换：*
- `transpose(dim0, dim1)`：返回转置后的新 TensorGeometry

*内部状态管理：*
- `mutable_sizes()`, `mutable_strides()`, `mutable_storage_offset()`：可变访问器
- `recompute()`：修改后重新计算 numel 和符号标志

---

### 要点总结

- 轻量级几何表示，用于避免完整张量构造的开销
- 支持符号尺寸（用于编译时形状推理）和具体整数尺寸
- 核心算法：从末维向前遍历验证连续性
- 提供只读和可变访问接口
- 缓存 `has_symbolic_sizes_strides_` 标志优化性能
