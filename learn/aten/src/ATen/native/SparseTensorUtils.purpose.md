# SparseTensorUtils 核心功能分析

## 文件用途
提供稀疏张量操作的底层工具函数，主要用于索引转换和稀疏张量结构操作。

## 主要功能模块

### 1. 索引扁平化 (Flatten Indices)

**flatten_indices** (SparseTensorUtils.cpp:37)
- 将多维稀疏索引转换为一维索引
- 示例：`[[2,4,0], [3,1,10]]` + `[2,12]` → `[27,49,10]`
- 计算方式：`2*12+3=27, 4*12+1=49, 0*12+10=10`
- 单维度时直接squeeze，多维度时调用stub函数
- `force_clone` 参数控制是否强制克隆结果

**flatten_indices_by_dims** (SparseTensorUtils.cpp:73)
- 支持部分维度扁平化
- 只处理指定维度列表，结果可能未合并(uncoalesced)
- 循环累乘累加：`new_indices *= sizes[d]` 后 `+= indices[d]`

### 2. 稀疏格式转换

**coo_to_csr** (SparseTensorUtils.cpp:82)
- COO格式行索引转换为CSR格式压缩行数组
- 输入：行指针数组、维度、非零元素数
- 输出：长度为 `dim+1` 的CSR数组
- 使用 `parallel_for` 并行处理，粒度10000

### 3. 稀疏张量构造

**zeros_like_with_indices** (SparseTensorUtils.cpp:114)
- 创建与输入相同索引结构但值全为0的稀疏张量
- 克隆原始索引，值用0填充后expand

**full_coo_indices** (SparseTensorUtils.cpp:126)
- 生成给定形状的全索引张量
- 等价于 `torch.ones(shape).nonzero().transpose(-2,-1)` 但更快
- 通过arange+unsqueeze+expand+stack实现

### 4. 底层访问器 (Header)

**get_sparse_impl** (SparseTensorUtils.h:27)
- 获取 `SparseTensorImpl*` 指针，用于访问内部字段

**alias_into_sparse** (SparseTensorUtils.h:35)
- 直接将索引和值放入稀疏张量，无拷贝

**copy_into_sparse** (SparseTensorUtils.h:44)
- 拷贝索引和值到稀疏张量

### 5. 辅助工具

**is_same_tensor** (SparseTensorUtils.h:56) - 比较两个张量是否同一底层实现
**is_same_density** (SparseTensorUtils.h:60) - 检查稀疏/密集维度是否相同
**new_values_with_size_of** (SparseTensorUtils.h:69) - 创建新非零元素数的values张量

### 6. TensorGeometryHolder 模板类

(SparseTensorUtils.h:124-182)
- 存储张量的几何信息(sizes/strides)
- 特化版本 `<static_shape_max_len>`: 用数组存储，适合固定小尺寸
- 特化版本 `<0>`: 用Tensor存储，适合动态大尺寸
- 根据维度数选择CPU或设备内存

## 关键设计特点

- **性能优化**: 并行处理、避免不必要拷贝、使用stub分发机制
- **灵活性**: 支持完整/部分扁平化、clone控制、维度选择
- **内存管理**: 区分alias/copy语义、几何信息存储优化

---

**Backward相关**: 无  
**ROCm相关**: 无
