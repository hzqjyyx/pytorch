这个文件提供了稀疏压缩张量（Sparse Compressed Tensor）的工具函数和宏定义，主要支持四种布局格式：CSR、CSC、BSR、BSC。

## 核心功能

### 1. 布局分发宏（Lines 16-138）
提供了一系列 `AT_DISPATCH_*` 宏，用于根据不同的稀疏压缩布局执行相应的代码分支：

- `AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS`: 处理所有四种布局
- `AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS`: 区分行压缩（CSR/BSR）和列压缩（CSC/BSC）
- `AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS`: 区分非块状（CSR/CSC）和块状（BSR/BSC）
- `AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS`: 仅处理非块状布局
- `AT_DISPATCH_SPARSE_COMPRESSED_BLOCK_LAYOUTS`: 仅处理块状布局

### 2. 布局判断和转换（Lines 172-211）
- `is_sparse_compressed()`: 判断是否为稀疏压缩布局
- `layoutToString()`: 将布局枚举转换为字符串表示
- `flip_compressed_layout()`: 在行列压缩布局之间转换（CSR↔CSC, BSR↔BSC）

### 3. 索引和维度操作（Lines 213-314）
- `isCompressedRow()/isCompressedColumn()`: 判断压缩维度方向
- `compressedIndicesName()/plainIndicesName()`: 获取索引名称（crow_indices/col_indices 等）
- `compressedDimName()/plainDimName()`: 获取维度名称（row/column/row block/column block）
- `rowDimension()/columnDimension()`: 计算行列维度索引
- `numBatchDimensions()`: 获取批次维度数量
- `getCompressedPlainIndices()`: 提取压缩索引和普通索引对

### 4. 张量属性访问（Lines 188-359）
- `get_sparse_csr_impl()`: 获取底层 SparseCsrTensorImpl 指针
- `getIndexDtype()`: 获取索引数据类型
- `getBlockSize()/getSymIntBlockSize()`: 获取块状布局的块大小

### 5. 二元操作优化（Lines 361-409）
`only_sparse_compressed_binary_op_trivial_cases()` 和 `only_sparse_compressed_add_trivial_cases()` 处理特殊情况：
- 当 self、other、out 是同一个张量时，直接操作 values
- 当 self 和 other 相同时，复用索引结构

### 6. 类型转换和缓冲区管理（Lines 411-452）
- `to_type()`: 转换张量数据类型，保持索引结构不变
- `create_acc_buffer()`: 创建累加缓冲区，支持混合精度计算
- `copy_from_acc_buffer()`: 从累加缓冲区复制回结果

### 7. RAII 不变量检查（Lines 149-168）
`CheckSparseTensorInvariants` 类用于临时启用/禁用稀疏张量不变量检查，确保异常安全。

## 设计特点
- 使用宏和模板实现零开销抽象
- 通过 lambda 表达式实现布局相关的代码分发
- 支持批次维度和块状稀疏格式
- 提供了统一的接口处理四种不同的压缩格式

---

**ROCm/Backward 相关内容：**
- 无直接 ROCm 相关代码
- 无 Backward 相关代码（这是纯工具函数文件）
