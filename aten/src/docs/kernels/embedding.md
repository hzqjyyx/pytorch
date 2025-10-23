# `embedding` 内核深度剖析（前向传播）

本文档详细介绍了 `torch.nn.Embedding` 前向传播在 PyTorch 中的实现过程。我们将从高级 Python API 开始，深入追踪到 CPU 和 CUDA 设备专用的 C++ 内核，重点关注数据查找的执行机制。

本文档基于 2025 年 9 月版本的 PyTorch 代码库。

## 1. 简介：`embedding` 作为查找操作

`embedding` 操作本质上是一个查找表。给定一个索引张量（通常是 `LongTensor`），它从权重矩阵（"嵌入"矩阵）中检索对应的向量。与"视图"操作（如 `transpose`）不同，嵌入是一个**数据复制操作**。它创建一个新的输出张量，并用从权重矩阵复制的数据填充它。

Python 中的典型调用如下所示：

```python
# 为10个项目创建一个3维向量的嵌入层
embedding_layer = torch.nn.Embedding(num_embeddings=10, embedding_dim=3)

# 创建一批要查找的索引
input_indices = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

# 执行查找
output = embedding_layer(input_indices)
```

这个简单的调用启动了一连串事件，最终导致高度优化的设备专用内核执行。

## 2. Python 调用栈

这个过程从面向用户的 Python API 开始，经过几个层次后到达 C++ 后端。

### 步骤 1：`nn.Module`
- **文件**: `torch/nn/modules/sparse.py`
- **行**: `15` (大约)
- **代码**: `class Embedding(Module):`
- **说明**: `torch.nn.Embedding` 类是一个有状态的 `nn.Module`。它的主要作用是初始化并保存 `weight` 参数，这是实际的嵌入矩阵。当模块实例被调用时（例如，`embedding_layer(...)`），它执行其 `forward` 方法。

### 步骤 2：函数式 API
- **文件**: `torch/nn/functional.py`
- **行**: `2551` (大约)
- **代码**: `return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)`
- **说明**: `Embedding` 模块的 `forward` 方法是对无状态的 `torch.nn.functional.embedding` 函数的封装。这是 PyTorch 的常见模式，将状态管理（`nn.Module`）与核心逻辑（`nn.functional`）分离。这个函数在一些验证后，进行关键调用，跨越边界进入 C++ 世界。

### 步骤 3：C++ 边界
- **代码**: `torch.embedding(...)`
- **说明**: 这个调用调用了一个通过 pybind11 库暴露给 Python 的 C++ 函数。这是进入 PyTorch ATen 库的入口点。

## 3. C++ 分发机制

一旦进入 C++，调用不会直接到单个函数，而是到**ATen 分发器**。

- **文件**: `aten/src/ATen/native/native_functions.yaml`
- **行**: `2292` (大约)
- **代码**: `- func: embedding(Tensor weight, Tensor indices, ...)`
- **说明**: 这个 YAML 文件声明性地定义了所有原生 ATen 函数的签名。分发器读取这些信息，根据输入张量（`weight` 和 `indices`）的 `Device`，将调用路由到正确的注册内核。如果张量在 CPU 上，它调用 CPU 内核；如果它们在 CUDA 设备上，它调用 CUDA 内核。

对于前向传播，`embedding` 函数本质上是另一个核心 ATen 算子的特殊形式：`index_select`。因此，以下章节将分析执行核心逻辑的 `index_select` 实现。

## 4. CPU 实现

当分发器选择 CPU 后端时，它最终调用针对 CPU 执行优化的内核。

- **文件**: `aten/src/ATen/native/cpu/Indexing.cpp`

CPU 上 `index_select` 的核心逻辑在 `index_select_kernel_impl` 函数中找到。

```cpp
// 在 aten/src/ATen/native/cpu/Indexing.cpp 中

static void index_select_kernel_impl(
    char* C_data, const int64_t C_stride,
    const char* A_data, const int64_t A_stride,
    const int64_t* B_data, const int64_t B_size,
    const int64_t slice_size) {
  at::parallel_for(0, B_size, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      const int64_t index = B_data[i];
      // C_data 是输出张量数据
      // A_data 是权重矩阵数据
      char* C_ptr = C_data + i * C_stride;
      const char* A_ptr = A_data + index * A_stride;
      memcpy(C_ptr, A_ptr, slice_size);
    }
  });
}
```

**说明：**
1.  **`parallel_for`**: 操作在要查找的索引数量上并行化。`at::parallel_for` 使用线程池（如 OpenMP）将工作分配给多个 CPU 核心。
2.  **迭代**: 每个线程遍历 `indices` 张量的一个子集（由 `B_data` 表示）。
3.  **地址计算**: 对于每个索引，它计算权重矩阵中的源指针（`A_ptr`）和输出张量中的目标指针（`C_ptr`）。这通过张量的步长完成。
4.  **`memcpy`**: `memcpy` 函数执行实际的数据复制，将一个嵌入向量的字节从权重矩阵移动到输出张量。

## 5. CUDA 实现

当张量在 GPU 上时，分发器调用 CUDA 内核，这是一个设计用于大规模并行执行的函数。

- **文件**: `aten/src/ATen/native/cuda/Indexing.cu`

`index_select` 的 CUDA 实现使用 `index_select_kernel` 函数。

```cu
// 在 aten/src/ATen/native/cuda/Indexing.cu 中

template <typename scalar_t, typename index_t>
__global__ void index_select_kernel(
    TensorInfo<scalar_t, index_t> dst,
    TensorInfo<scalar_t, index_t> src,
    TensorInfo<index_t, index_t> indices,
    int64_t inner_size) {

  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < indices.sizes[0];
       i += blockDim.x * gridDim.x) {

    // 每个线程从索引张量获取一个索引
    index_t index = indices.data[i];
    
    // 获取源（权重）和目标（输出）的指针
    scalar_t* dst_ptr = dst.data + i * dst.strides[0];
    scalar_t* src_ptr = src.data + index * src.strides[0];

    // 逐元素复制嵌入向量
    for (int64_t j = 0; j < inner_size; ++j) {
      dst_ptr[j] = src_ptr[j];
    }
  }
}
```

**说明：**
1.  **`__global__`**: 这个关键字标记 `index_select_kernel` 为在 GPU 上运行并可以从 CPU 调用的函数。
2.  **网格跨步循环**: 内核以线程块网格的形式启动。`for` 循环是 CUDA 中标准的"网格跨步循环"模式，允许固定数量的线程处理任意大量的索引。
3.  **并行查找**: 每个 GPU 线程独立地从 `indices` 张量读取一个索引。
4.  **并行复制**: 每个线程将一个完整的嵌入向量从源（全局 GPU 内存中的 `weight` 张量）复制到目标（也在全局 GPU 内存中的输出张量）。因为成千上万个线程同时执行这个操作，总体操作极其快速。

## 6. 总结：CPU vs CUDA 前向传播

| 特性 | CPU 实现（`index_select_kernel_impl`） | CUDA 实现（`index_select_kernel`） |
|---|---|---|
| **文件** | `aten/src/ATen/native/cpu/Indexing.cpp` | `aten/src/ATen/native/cuda/Indexing.cu` |
| **执行模型** | 通过多核线程并行（`at::parallel_for`）。 | 通过成千上万个 GPU 线程大规模并行。 |
| **核心操作** | `memcpy` 在循环中复制整个向量。 | 每线程循环复制向量元素。 |
| **粒度** | 一个 CPU 线程处理一批索引。 | 一个 GPU 线程通常一次处理一个索引。 |

## 7. 结论

`embedding` 前向传播清楚地说明了 PyTorch 的核心设计理念。一个简单的 Python API 调用通过抽象层转换为设备专用的高度优化内核。分发器干净地将通用逻辑与硬件专用实现分离，允许相同的用户代码在不同后端上高效运行。使用 `index_select` 作为底层机制展示了复杂操作通常是如何由 ATen 库中强大的基础原语组合而成的。
