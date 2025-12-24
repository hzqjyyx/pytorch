# Indexing.cu - CUDA 索引操作

这是 **PyTorch 的 ATen 库中 CUDA 索引操作的实现文件** ，位置 `@aten/src/ATen/native/cuda/Indexing.cu`（共 2080 行代码）。

## 核心功能模块

### 1. **索引赋值（Index Put）操作**
- `index_put_with_sort_kernel()`：将值赋给指定索引位置，支持累加模式
- `index_put_with_sort_quantized()`：支持量化张量的索引赋值
- 使用排序+去重策略处理重复索引问题

### 2. **索引反向传播（Backward）**
- `indexing_backward_kernel()` 的多个变体（支持不同步长和数据类型）
- `indexing_backward_kernel_stride_1()`：优化步长为 1 的情况
- `indexing_backward_kernel_quantized()`：量化张量的梯度计算
- 处理索引操作的梯度计算

### 3. **索引选择（Index Select）**
- `index_select_out_cuda_impl()`：从张量中按索引选择元素
- `indexSelectSmallIndex()` 和 `indexSelectLargeIndex()`：根据索引范围优化的 CUDA 核函数
- `index_select_sparse_cuda()`：稀疏张量的索引选择操作

### 4. **索引加法（Index Add）**
- `index_add_cuda_impl()`：在指定索引位置累加值

### 5. **索引规约（Index Reduce）**
- `index_reduce_func_cuda_impl()`：对指定索引位置执行规约操作（如求和、最大值等）

### 6. **掩码填充（Masked Fill）**
- `masked_fill_kernel()`：根据掩码填充张量指定位置的值
- `masked_fill_kernel_quantized()`：量化张量的掩码填充

### 7. **稀疏张量索引**
- `index_select_sparse_cuda()`：处理稀疏 COO 张量的索引操作

## 关键特性

- **GPU 加速**：所有操作都在 CUDA 上执行
- **量化支持**：通过 `index_put_with_sort_quantized()` 和 `masked_fill_kernel_quantized()` 实现量化张量的索引操作
- **性能优化**：根据索引范围大小提供 `indexSelectSmallIndex()` 和 `indexSelectLargeIndex()` 两个优化核函数版本
- **梯度支持**：提供 `indexing_backward_kernel_stride_1()` 等多个反向传播内核，针对不同步长优化
- **稀疏张量**：支持稀疏 COO 张量操作，独立处理稀疏维度和密集维度
- **ROCM 支持**：代码包含针对 ROCM（AMD GPU）的优化分支，如增大默认线程数（512 vs 128）和共享内存预分配

## 应用场景

主要用于 PyTorch 中的以下操作：
  tensor[indices] = value        # Index put
  tensor[indices] += value       # Index add
  output = tensor[indices]       # Index select
  tensor[mask] = value           # Masked fill