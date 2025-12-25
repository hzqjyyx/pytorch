# ScatterGatherKernel.cu 主要功能分析

这个文件实现了 CUDA 版本的 scatter/gather 操作及其变体的核心内核代码。

## 核心操作类型

文件定义了多个 reduction 操作的 functor：

- **TensorAssign** (lines 69-77): 直接赋值 `self[index] = src`
- **ReduceAdd** (lines 31-38): 原子加法，用于 scatter_add
- **ReduceMultiply** (lines 21-29): 原子乘法
- **ReduceMinimum** (lines 49-57): 原子最小值
- **ReduceMaximum** (lines 59-67): 原子最大值  
- **ReduceMean** (lines 40-47): 均值（实现与 ReduceAdd 相同）

## 内核架构

**底层执行引擎** (lines 86-113):
- `_scatter_gather_elementwise_kernel`: 实际执行的 CUDA kernel，使用线程块并行处理
- `_launch_scatter_gather_kernel`: kernel 启动器，配置 grid/block 维度

**类型抽象** (line 83):
- `OpaqueType<N>`: 不透明类型，避免为相同大小的不同类型生成冗余内核

## 三个核心实现模板

### 1. cuda_scatter_gather_base_kernel (lines 159-335)

处理 scatter/gather 操作，根据 `is_scatter_like` 模板参数区分：

**Scatter 模式** (`is_scatter_like=true`):
```
self[index[i]] = src[i]  (或使用 reduction op)
```

**Gather 模式** (`is_scatter_like=false`):
```
result[i] = src[index[i]]
```

核心逻辑：
- 重新调整 stride，使 `self.shape = src.shape = index.shape`
- 对于 scatter，设置 `self.stride[dim] = 0`；对于 gather，设置 `src.stride[dim] = 0`
- 使用 `TensorIterator` 遍历所有元素
- 调用 `_cuda_scatter_gather_internal_kernel` (lines 116-157) 执行实际计算

### 2. cuda_scatter_fill_base_kernel (lines 381-469)

处理 scatter_fill 操作（用标量填充）：
```
self[index[i]] = scalar_value
```

与 `cuda_scatter_gather_base_kernel` 的区别：
- 源数据是标量而非张量
- 只需处理 `self` 和 `index`，不涉及 `src` 张量
- 使用 `_cuda_scatter_fill_internal_kernel` (lines 337-379)

### 3. 索引边界检查

在 `_cuda_scatter_gather_internal_kernel` (line 144) 和 `_cuda_scatter_fill_internal_kernel` (line 365) 中：
```cuda
CUDA_KERNEL_ASSERT(idx_dim >= 0 && idx_dim < index_size && "index out of bounds");
```

## 对外接口函数

文件提供七个注册到 dispatcher 的函数：

1. **gather_cuda_kernel** (lines 471-475): 基础 gather 操作
2. **scatter_cuda_kernel** (lines 477-483): 基础 scatter 操作
3. **scatter_fill_cuda_kernel** (lines 485-489): 用标量填充
4. **scatter_add_cuda_kernel** (lines 491-498): scatter 累加
5. **scatter_reduce_cuda_kernel** (lines 500-517): scatter reduce (支持 SUM/PROD)
6. **scatter_reduce_two_cuda_kernel** (lines 519-546): 扩展版 reduce (支持 SUM/PROD/MAX/MIN/MEAN)
7. **scatter_scalar_reduce_cuda_kernel** (lines 548-562): 标量版 reduce

## 数据类型支持

通过 `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3` 宏支持：
- 所有基础类型（int, float, double 等）
- 复数类型
- Half, BFloat16, Bool

## 特殊处理

**32-bit 索引限制** (lines 126-133, 348-355):
- 如果 tensor 无法使用 32-bit 索引，自动分割成多个子 iterator

**非确定性警告** (lines 479, 494, 504, 523, 528, 541):
- 由于使用原子操作，当索引不唯一时结果是非确定性的
- 会调用 `globalContext().alertNotDeterministic()` 发出警告

---

**ROCm/Backward 相关内容**:
- 无 ROCm 特定代码
- 无反向传播相关实现（这是前向操作的底层内核）
