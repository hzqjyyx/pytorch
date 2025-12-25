# SparseCsrTensorImpl 核心功能

## 数据结构设计

实现 PyTorch 的稀疏 CSR（Compressed Sparse Row）张量的底层表示，使用三个 1-D 张量存储数据：

- `crow_indices_`: 压缩行索引，形状 `(size(0) + 1)`，存储每行非零元素在 values 中的起始位置
- `col_indices_`: 列索引，形状 `(nnz)`，存储每个非零元素的列位置  
- `values_`: 值张量，形状 `(nnz)`，存储实际的非零元素值

支持多种布局变体：`kSparseCsr`、`kSparseCsc`、`kSparseBsr`、`kSparseBsc`

## 构造函数

### 基础构造函数 (SparseCsrTensorImpl.cpp:11-38)
接收 DispatchKeySet、Device、Layout 和数据类型，创建空的索引和值张量（默认大小为 0）

### 完整构造函数 (SparseCsrTensorImpl.cpp:40-80)
接收已有的 crow_indices、col_indices、values 张量，执行关键验证：
- 检查 DispatchKey 与设备类型一致性（CPU/CUDA/XPU/Meta/PrivateUse1）
- 确保三个成员张量在同一设备上
- 设置 `storage_access_should_throw()`，禁止直接访问底层存储
- 标记为非重叠非稠密（`is_non_overlapping_and_dense_ = false`）
- 发出 beta 功能警告

## 尺寸调整操作

### resize_ (SparseCsrTensorImpl.cpp:86-108)
调整稀疏张量到指定 nnz（非零元素数）和形状：
- 计算新的 crow_indices 大小为 `batch_dims + [rows + 1]`
- 如果行数增加，用 nnz 填充新增的 crow_indices 条目
- 如果行数减少，设置最后一个 crow_indices 为 `min(nnz, rows*cols)`
- 同步调整 col_indices 和 values 的大小

### resize_and_clear_ (SparseCsrTensorImpl.cpp:110-157)
按稀疏维度和稠密维度重新初始化：
- 验证稀疏维度必须为 2
- 根据布局类型（行压缩 vs 列压缩）计算压缩维度
- 处理块稀疏张量（BSR/BSC）的块大小，确保压缩维度可被块大小整除
- 将 crow_indices 清零，col_indices 和 values 调整为空（nse=0）

### resize_as_sparse_compressed_tensor_ (SparseCsrTensorImpl.cpp:159-195)
按源张量调整大小：
- 要求源张量与当前张量布局相同
- 复用现有索引存储，仅在尺寸不匹配时调整
- 如果整体形状或稠密维度改变，复制源张量的索引数据以保持不变性

## 成员张量设置

### set_member_tensors (SparseCsrTensorImpl.cpp:197-238)
设置三个核心成员张量并更新形状：
- 验证 values 的数据类型与张量声明的 dtype 一致
- 更新 `sizes_and_strides_` 并刷新 numel
- 检查所有成员张量在同一设备上

支持 `SymIntArrayRef` 和 `IntArrayRef` 两种尺寸参数

## 维度计算 (SparseCsrTensorImpl.h:69-84)

```cpp
batch_dim()  = crow_indices_.dim() - 1      // 批次维度数
sparse_dim() = 2                             // 稀疏维度固定为 2
dense_dim()  = values_.dim() - batch_dim() - block_dim() - 1
block_dim()  = (layout_ == kSparseBsr/Bsc) ? 2 : 0
```

## 不支持的操作 (SparseCsrTensorImpl.cpp:240-257)

稀疏压缩张量显式禁止以下操作，调用时抛出错误：
- `strides_custom()` / `sym_strides_custom()`: 无 strides 概念
- `set_size()` / `set_stride()` / `set_storage_offset()`: 不支持直接修改
- `is_contiguous_custom()`: 无连续性概念

## 浅拷贝与分离 (SparseCsrTensorImpl.h:111-168)

`shallow_copy_and_detach_core()` 处理两种情况：
1. **Python 模式激活**：通过 PyInterpreter 分离
2. **纯 C++ 模式**：创建新 SparseCsrTensorImpl，复制元数据但共享底层数据

元数据复制通过 `copy_tensor_metadata()` 完成，包括尺寸、版本计数器和三个成员张量的浅引用

---

**ROCm 相关**: 支持 `kSparseCsrXPU` DispatchKey，需与 XPU 设备类型匹配

**Backward 相关**: 无直接自动微分逻辑，梯度计算通过 dispatcher 路由到对应内核
