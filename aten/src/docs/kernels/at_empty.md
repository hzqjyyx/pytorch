# `at::empty` 实现深度剖析

本文档详细介绍了 `at::empty` 函数在 PyTorch 的 C++ 后端 ATen 中的实现过程。我们将追踪 CPU 和 CUDA 设备的执行路径，探索创建新的未初始化张量所涉及的关键组件和设计原则。

本文档基于 2025 年 9 月版本的 PyTorch 代码库。

## 1. 简介：`at::empty` 的作用

`at::empty` 本质上是一个工厂函数。它的主要目的是创建指定大小的新张量，但不初始化其数据。为张量分配的内存将包含该内存位置中已存在的任何数据。这使得 `at::empty` 成为创建张量的最快方式，如果您计划随后立即用数据填充它的话。

C++ 中典型的 `at::empty` 调用如下所示：

```cpp
// 在 CPU 上创建一个 2x3 的浮点张量
auto tensor = at::empty({2, 3}, at::kFloat);
```

这个看似简单的函数调用触发了一个复杂的分发机制，将请求路由到适当的后端（CPU、CUDA 等）来执行实际的内存分配和张量创建。

## 2. 函数调用的旅程：分发和命名空间

在深入具体实现之前，理解 PyTorch 如何将像 `at::empty` 这样的通用函数调用定向到正确的设备专用代码至关重要。这由**分发器**处理。

正如 `aten_overview.md` 中所概述，ATen 使用声明式方法在 `aten/src/ATen/native/native_functions.yaml` 中定义操作符。这个文件作为单一的真相源。当您调用 `at::empty` 时，分发器会查看 `TensorOptions` 中指定的 `Device`（或默认设备，如果没有提供）。根据这个设备，它选择适当的内核来执行。

这个旅程涉及几个命名空间，每个都有不同的目的：

*   **`at::`**: 这是 ATen C++ API 面向公众的命名空间。这个命名空间中的函数（如 `at::empty`）是使用 PyTorch C++ API 的开发者应该调用的。它们通常是轻量级的包装器，将工作委托给分发器。

*   **`at::native::`**: 这个命名空间包含不同后端操作符的“原生”实现。这些是分发器调用的函数。您将在这里找到像 `at::native::empty_cpu` 和 `at::native::empty_cuda` 这样的函数。它们是设备专用逻辑的入口点。

*   **`at::detail::`**: 这个命名空间用于不属于公共或原生 API 的实现细节。这些函数不意在被用户直接调用，而是在内部用于共享代码和实现张量创建的低级逻辑。

现在，让我们追踪 CPU 和 CUDA 的完整执行路径。

## 3. `at::empty` 的 CPU 实现

在 CPU 上创建张量是最基本的情况。调用栈如下：

`at::empty` -> `at::native::empty_cpu` -> `at::detail::empty_cpu` -> `at::detail::empty_generic`

让我们逐步检查每个步骤。

### 步骤 1：公共 API - `at::empty`

旅程从公共的 `at::empty` 函数开始。虽然它的确切定义是从 `native_functions.yaml` 生成的，但它实际上充当分发器，对于 CPU 设备，将导向已注册的 CPU 内核。

### 步骤 2：原生内核 - `at::native::empty_cpu`

*   **文件：** `aten/src/ATen/native/TensorFactories.cpp`

分发器调用 `at::native::empty_cpu`，这是 `CPU` 分发键的已注册内核。这个函数是通向更详细实现的垫脚石。

```cpp
// aten/src/ATen/native/TensorFactories.cpp

Tensor empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  Tensor result = at::detail::empty_cpu(
      size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      memory_format_opt);
  // ... (可选的确定性填充) ...
  return result;
}
```

这个函数的主要作用是将工作委托给 `at::detail::empty_cpu`。

### 步骤 3：细节实现 - `at::detail::empty_cpu`

*   **文件：** `aten/src/ATen/EmptyTensor.cpp`

这个函数处理 CPU 专用的细节，主要是选择正确的内存分配器。

```cpp
// aten/src/ATen/EmptyTensor.cpp

TensorBase empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    // ... 其他选项 ...
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  // ...
  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);
}

TensorBase empty_cpu(IntArrayRef size, ScalarType dtype, bool pin_memory,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}
```

这里发生的事情：
1.  它解析 `pin_memory` 选项。如果 `pin_memory` 为 `true`，`GetCPUAllocatorMaybePinned` 返回一个特殊的“钉住”内存分配器，该分配器分配页锁定的主机内存，用于更快的 CPU 到 GPU 数据传输。
2.  如果 `pin_memory` 为 `false`，它返回标准的 `c10::GetCPUAllocator()`。
3.  然后它调用最终的通用实现 `empty_generic`，传递选定的分配器。

### 步骤 4：核心逻辑 - `at::detail::empty_generic`

*   **文件：** `aten/src/ATen/EmptyTensor.cpp`

这里是奇迹发生的地方。这个函数与后端无关；它以分配器作为参数，不在乎它是 CPU、CUDA 还是任何其他分配器。

```cpp
// aten/src/ATen/EmptyTensor.cpp

TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  
  // 1. 计算存储大小
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());

  // 2. 分配内存并创建 StorageImpl
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  // 3. 创建 TensorImpl
  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
      
  // 4. 设置张量属性
  tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);

  if (memory_format_opt.has_value()) {
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  return tensor;
}
```

这个过程涉及三个关键对象：

1.  **`Storage` 和 `StorageImpl`**: `Storage` 对象表示单个连续的内存块。`StorageImpl` 是实际的实现，它保持一个 `DataPtr`，这是一个指向由提供的 `allocator` 分配的原始内存的智能指针。`empty_generic` 计算所需的字节数并创建该大小的 `StorageImpl`。

2.  **`Tensor` 和 `TensorImpl`**: 用户交互的 `at::Tensor` 是一个指向 `TensorImpl` 的轻量级句柄（智能指针）。`TensorImpl` 是真正表示张量的对象。它保持对 `StorageImpl` 的引用，并包含像大小、步长和数据类型等元数据。`empty_generic` 创建一个 `TensorImpl` 并将其链接到新创建的 `StorageImpl`。

3.  **设置元数据**: 最后，函数在新的 `TensorImpl` 上设置 `sizes` 和 `strides`。如果请求特定的 `memory_format`（如 `ChannelsLast`），它会相应地调整步长。

## 4. `at::empty` 的 CUDA 实现

创建 CUDA 张量的过程非常相似，这证明了 ATen 模块化设计的威力。

调用栈是：
`at::empty` -> `at::native::empty_cuda` -> `at::detail::empty_cuda` -> `at::detail::empty_generic`

### 步骤 1 和 2：公共 API 和原生内核

*   **文件：** `aten/src/ATen/native/cuda/TensorFactories.cu`

就像 CPU 情况一样，带有 CUDA 设备选项的 `at::empty` 调用被分发到 `at::native::empty_cuda`。

```cu
// aten/src/ATen/native/cuda/TensorFactories.cu

Tensor empty_cuda(IntArrayRef size, std::optional<ScalarType> dtype_opt, /*...其他选项...*/) {
  Tensor result = at::detail::empty_cuda(size, dtype_opt, /*...其他选项...*/);
  // ... (可选的确定性填充) ...
  return result;
}
```

### 步骤 3：细节实现 - `at::detail::empty_cuda`

*   **文件：** `aten/src/ATen/cuda/EmptyTensor.cpp`

这里是主要差异所在。这个函数明确检索 CUDA 分配器，而不是 CPU 分配器。

```cpp
// aten/src/ATen/cuda/EmptyTensor.cpp

TensorBase empty_cuda(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<Device> device_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
      
  // 1. 初始化 CUDA 并设置设备
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  const auto device = device_or_default(device_opt);
  const DeviceGuard device_guard(device);
  
  // 2. 获取 CUDA 分配器
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  
  // 3. 调用通用实现
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_generic(
      size, allocator, cuda_dks, dtype, memory_format_opt);
}
```
这个函数确保为正确的设备初始化 CUDA 上下文，然后获取 `CUDADeviceAllocator`。这个分配器负责通过 `cudaMalloc` 和 `cudaFree` 管理 GPU 上的内存。

### 步骤 4：核心逻辑 - `at::detail::empty_generic`

最后一步再次是调用 `aten/src/ATen/EmptyTensor.cpp` 中**完全相同**的 `at::detail::empty_generic` 函数。这个函数的执行过程与 CPU 情况完全相同，但这次它接收的 `allocator` 是 `CUDADeviceAllocator`。当创建 `storage_impl` 时，它调用分配器的 `allocate` 方法，现在的结果是在 GPU 上分配内存。

## 5. 总结：CPU vs. CUDA 实现

| 特性 | CPU 实现 | CUDA 实现 |
| --------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **原生入口点** | `at::native::empty_cpu` | `at::native::empty_cuda` |
| **文件位置** | `aten/src/ATen/native/TensorFactories.cpp` | `aten/src/ATen/native/cuda/TensorFactories.cu` |
| **细节函数** | `at::detail::empty_cpu` | `at::detail::empty_cuda` |
| **文件位置** | `aten/src/ATen/EmptyTensor.cpp` | `aten/src/ATen/cuda/EmptyTensor.cpp` |
| **分配器** | `c10::GetCPUAllocator()` 或钉住内存分配器。 | `at::cuda::getCUDADeviceAllocator()` |
| **核心逻辑** | `aten/src/ATen/EmptyTensor.cpp` 中的 `at::detail::empty_generic` | `aten/src/ATen/EmptyTensor.cpp` 中的 `at::detail::empty_generic` |

## 6. 结论

`at::empty` 的实现是 PyTorch 后端中抽象和模块化设计威力的完美例子。通过分离公共 API（`at::`）、原生分发入口点（`at::native::`）和低级细节（`at::detail::`），系统实现了几个关键目标：

*   **代码可重用性**: 创建张量元数据（`TensorImpl`、`StorageImpl`）的核心逻辑在 `empty_generic` 中在所有后端之间共享。
*   **可扩展性**: 添加新后端（例如，用于新类型加速器）只需要提供新的原生内核（例如，`empty_my_backend`）和相应的分配器。核心逻辑保持不变。
*   **清晰性**: 关注点的分离使代码库更易于导航。如果您想了解 CUDA 分配的具体细节，您可以查看 `at::detail::empty_cuda`。如果您想了解从存储创建张量的通用过程，您可以查看 `at::detail::empty_generic`。

这种分层和分发的方法是 PyTorch 运行的基础，使它能够高效地支持广泛的硬件和张量类型，同时保持清洁一致的 API。
