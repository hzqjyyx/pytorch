## NestedTensor 核心概念

NestedTensor 是一种通用的张量数据结构，用于处理不同形状和长度的数据。它将多个张量（tensor_component）扁平化并合并为单个 NestedTensor，同时保留重构原始张量所需的元数据。

### 数据结构组成
- **nested_sizes_**: n个张量 × n维 的2D张量
- **nested_strides_**: n个张量 × n维 的2D张量  
- **storage_offsets_**: 每个张量分量起始位置的偏移量
- **storage_**: 包含扁平化张量分量的存储对象

### 代码组织

- `NestedTensorImpl.h/cpp` - 数据结构定义和方法实现
- `NestedTensorUtils.h/cpp` - 工具函数（包含 `map_nested_tensor`）
- `NestedTensorUnaryOps.cpp` - 一元操作（abs, log, relu 等）
- `NestedTensorBinaryOps.h/cpp` - 二元操作（add, mul, div 等）
- `NestedTensorFactories.cpp` - 张量创建函数（empty_like 等）
- `NestedTensorMath.h/cpp` - 数学函数（softmax, embedding 等）
- `NestedTensorMatmul.cpp` - 矩阵操作（matmul, linear, bmm 等）
- `NestedTensorTransformerFunctions.h/cpp` - Transformer 相关函数
- `cuda/` - CUDA 实现

### 两类函数实现策略

**第一类**：可通过将 NestedTensor 视为密集张量高效实现
- 使用 `map_nt` 模板函数处理一元/二元运算
- 流程：提取 impl → 获取 sizes → 调用函数 f → 构造新 NestedTensor

**第二类**：无法通过密集视图高效实现（如不规则维度的 softmax）
- 使用 `map_nested_tensor` 逐个处理张量分量
- CPU 实现的良好基线，但 CUDA 需要单独优化

### 关键要点

- NestedTensor 继承自 `c10::TensorImpl`
- 构造时使用 `wrap_buffer` 函数确保输入张量连续性
- 实现新运算符时需添加单元测试（test/test_nestedtensor.py）
- 第二类函数的核心挑战是避免为每个张量分量启动单独的 CUDA 核
