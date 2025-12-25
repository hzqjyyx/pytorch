# EmbeddingBag 核心功能

这些文件实现了 PyTorch 的 `embedding_bag` 操作，用于高效地从嵌入表中查找并聚合多个嵌入向量。

## 核心概念

**EmbeddingBag** 将传统的 embedding lookup + reduction 两步操作融合为一步：
- 从权重矩阵 `weight[vocab_size × embed_dim]` 中根据 `indices` 查找嵌入向量
- 将属于同一"bag"的多个嵌入向量按指定模式聚合（SUM/MEAN/MAX）

**三种聚合模式** (EmbeddingBag.h:11-15):
- `SUM`: 直接求和
- `MEAN`: 求平均值
- `MAX`: 取最大值

## 核心数据结构

**Offsets 机制**：
- `indices`: 要查找的所有嵌入索引，如 `[0, 1, 2, 3, 4]`
- `offsets`: 每个 bag 的起始位置，如 `[0, 2, 5]` 表示 bag0=[0,1], bag1=[2,3,4]
- `offset2bag`: 反向映射，将每个 index 映射到其所属的 bag 编号

**offset2bag 构建** (EmbeddingBag.cpp:57-62):
```
输入: offsets = [0, 2, 5]
过程: 
  - index_add 得 [1, 0, 1, 0, 1]
  - 首位减1 得 [0, 0, 1, 0, 1]  
  - cumsum 得  [0, 0, 1, 1, 2]
结果: offset2bag[i] 表示 indices[i] 属于哪个 bag
```

## 快速路径优化

**is_fast_path** 判断条件 (EmbeddingBag.cpp:81-103):
- 数据类型为 float/half/bfloat16
- 内存连续（stride[1] == 1）
- 无 padding（padding_idx < 0）
- 满足条件时可使用 FBGEMM 或专门优化的代码路径

**FBGEMM 加速** (USE_FBGEMM 宏):
- 使用 `GenerateEmbeddingSpMDM` 生成优化的稀疏矩阵-密集矩阵乘法内核
- 支持 FP32/FP16/BF16 数据类型
- 通过 `_EmbeddingBagKernelCache` 缓存生成的内核函数，避免重复生成

## 主要实现路径

### 1. SUM/MEAN 模式

**index_select_add 函数族** (三个重载版本):

**Float 版本** (EmbeddingBag.cpp:368-488):
- **快速路径**: 使用 FBGEMM 的 `kernel_fp32_index_t` 或 Caffe2 的 `EmbeddingLookupIdx`
  - 并行处理每个 bag（`parallel_for` over output_size）
  - 直接从源数据读取并累加到输出
- **慢速路径**: 逐个索引处理
  - 使用 BLAS `axpy` 操作：`output[add_indices[i]] += weight[select_indices[i]]`
  - 遇到 padding_idx 时减少 bag_size 计数

**Half/BFloat16 版本** (EmbeddingBag.cpp:186-366):
- **快速路径**: 
  - 使用 FP32 中间缓冲区进行累加
  - FBGEMM 路径直接在 uint16 上操作
  - 非 FBGEMM 路径：FP32 累加后转回 FP16/BF16
- **慢速路径**: 
  - 转换到 FP32 → axpy → 转回原类型

**Double 版本** (EmbeddingBag.cpp:109-154):
- 仅慢速路径（无 FBGEMM 支持）
- 直接使用 double 精度 BLAS axpy

**per_sample_weights 支持** (index_select_scale_add):
- 在 SUM 模式下，每个嵌入向量乘以对应权重后再累加
- 快速路径：将权重传递给 FBGEMM kernel
- 慢速路径：`output[bag] += weight[idx] * scale[i]`

### 2. MAX 模式

**embedding_bag_cpu_max_out** (EmbeddingBag.cpp:1062-1124):
```cpp
for each index in indices:
    bag = offset2bag[index]
    if index != padding_idx:
        for each dimension:
            if first_in_bag or weight[index][dim] > output[bag][dim]:
                output[bag][dim] = weight[index][dim]
                max_indices[bag][dim] = index  // 记录最大值来源
```

## 主函数调用链

**embedding_bag** (EmbeddingBag.cpp:1225-1260):
1. 处理 padding_idx（支持负索引）
2. 判断是否需要梯度
3. 调用 `_embedding_bag_forward_only` 或 `_embedding_bag`

**_embedding_bag_cpu_impl** (EmbeddingBag.cpp:1176-1219):
1. `check_arguments`: 验证输入张量的形状和类型
2. 创建输出张量 `[num_bags × embed_dim]`
3. `make_offset2bag`: 构建索引到 bag 的映射
4. `make_bag_size`: 计算每个 bag 的大小
5. `make_max_indices`: 为 MAX 模式分配输出
6. `_embedding_bag_cpu_impl_out`: 执行实际计算

**_embedding_bag_cpu_impl_out** (EmbeddingBag.cpp:1126-1172):
- **SUM/MEAN**: 
  - 双重分发：`AT_DISPATCH_FLOATING_TYPES_AND2` × `AT_DISPATCH_INDEX_TYPES`
  - 调用 `index_select_add` 或 `index_select_scale_add`
  - MEAN 模式额外除以 bag_size (apply_bag_size)
- **MAX**: 
  - 调用 `embedding_bag_cpu_max_out`

## 辅助功能

**类型提升** (promoteIndicesAndOffsets, EmbeddingBag.cpp:66-76):
- 将 indices 和 offsets 提升到相同的整数类型（int32/int64）

**边界检查** (check_arguments, EmbeddingBag.cpp:868-913):
- offsets[0] 必须为 0
- offsets[-1] 不能超过 indices 长度
- per_sample_weights 仅支持 SUM 模式

**include_last_offset 处理**:
- 当为 true 时，offsets 的最后一个元素指定最后一个 bag 的结束位置
- 否则，最后一个 bag 延伸到 indices 末尾

## 性能优化技术

1. **内核缓存**: `_EmbeddingBagKernelCache` 缓存 FBGEMM 生成的函数指针
2. **并行化**: 使用 `at::parallel_for` 并行处理多个 bags
3. **向量化**: BF16 转换使用 SIMD 指令 (`vec::Vectorized`)
4. **内存布局优化**: 快速路径要求连续内存（stride[1]==1）
5. **索引排序**: backward 中对 indices 排序以提高缓存局部性

---

## ROCm 和 Backward 简要说明

- **Backward 操作**:
  - `_embedding_bag_backward`: 分发到 dense 或 sparse backward
  - `_embedding_bag_dense_backward`: 对 indices 排序后使用 BLAS axpy 累加梯度
  - `_embedding_bag_sparse_backward`: 返回稀疏梯度张量
  - `_embedding_bag_per_sample_weights_backward`: 计算 per_sample_weights 的梯度（点积）
  - MAX 模式使用 `max_indices` 记录的位置进行梯度反向传播
  
- **ROCm 支持**: 代码中未见 ROCm 特定路径，主要依赖 FBGEMM（x86）或 Caffe2 通用实现
