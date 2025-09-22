# A Deep Dive into the Implementation of `at::empty`

This document provides a detailed, educative walkthrough of how the `at::empty` function is implemented in PyTorch's C++ backend, ATen. We will trace the execution path for both CPU and CUDA devices, exploring the key components and design principles involved in creating a new, uninitialized tensor.

This explanation is based on the PyTorch codebase as of September 2025.

## 1. Introduction: The Role of `at::empty`

At its core, `at::empty` is a factory function. Its primary purpose is to create a new tensor of a specified size, but without initializing its data. The memory allocated for the tensor will contain whatever data was already present in that memory location. This makes `at::empty` the fastest way to create a tensor if you plan to fill it with data immediately afterward.

A typical call to `at::empty` in C++ looks like this:

```cpp
// Create a 2x3 tensor of floats on the CPU
auto tensor = at::empty({2, 3}, at::kFloat);
```

This seemingly simple function call triggers a sophisticated dispatch mechanism that routes the request to the appropriate backend (CPU, CUDA, etc.) to perform the actual memory allocation and tensor creation.

## 2. The Journey of a Function Call: Dispatch and Namespaces

Before diving into the specific implementations, it's crucial to understand how PyTorch directs a generic function call like `at::empty` to the correct device-specific code. This is handled by the **dispatcher**.

As outlined in the `aten_overview.md`, ATen uses a declarative approach for defining operators in `aten/src/ATen/native/native_functions.yaml`. This file acts as a single source of truth. When you call `at::empty`, the dispatcher looks at the `Device` specified in the `TensorOptions` (or the default device if none is provided). Based on this device, it selects the appropriate kernel to execute.

This journey involves several namespaces, each with a distinct purpose:

*   **`at::`**: This is the public-facing namespace for the ATen C++ API. Functions in this namespace, like `at::empty`, are what developers using the PyTorch C++ API are intended to call. They are generally lightweight wrappers that delegate to the dispatcher.

*   **`at::native::`**: This namespace contains the "native" implementations of operators for different backends. These are the functions that the dispatcher calls. You'll find functions like `at::native::empty_cpu` and `at::native::empty_cuda` here. They are the entry points for device-specific logic.

*   **`at::detail::`**: This namespace is for implementation details that are not part of the public or native API. These functions are not meant to be called directly by users but are used internally to share code and implement the lower-level logic of tensor creation.

Now, let's trace the complete execution path for both CPU and CUDA.

## 3. The CPU Implementation of `at::empty`

Creating a tensor on the CPU is the most fundamental case. The call stack is as follows:

`at::empty` -> `at::native::empty_cpu` -> `at::detail::empty_cpu` -> `at::detail::empty_generic`

Let's examine each step.

### Step 1: The Public API - `at::empty`

The journey begins with the public `at::empty` function. While its exact definition is generated from `native_functions.yaml`, it effectively acts as a dispatcher that, for a CPU device, will lead to the registered CPU kernel.

### Step 2: The Native Kernel - `at::native::empty_cpu`

*   **File:** `aten/src/ATen/native/TensorFactories.cpp`

The dispatcher calls `at::native::empty_cpu`, which is the registered kernel for the `CPU` dispatch key. This function is a stepping stone to the more detailed implementation.

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
  // ... (optional deterministic fill) ...
  return result;
}
```

This function's main role is to delegate the work to `at::detail::empty_cpu`.

### Step 3: The Detail Implementation - `at::detail::empty_cpu`

*   **File:** `aten/src/ATen/EmptyTensor.cpp`

This function handles the CPU-specific details, primarily selecting the correct memory allocator.

```cpp
// aten/src/ATen/EmptyTensor.cpp

TensorBase empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    // ... other options ...
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

Here's what happens:
1.  It resolves the `pin_memory` option. If `pin_memory` is `true`, `GetCPUAllocatorMaybePinned` returns a special "pinned" memory allocator that allocates page-locked host memory for faster CPU-to-GPU data transfers.
2.  If `pin_memory` is `false`, it returns the standard `c10::GetCPUAllocator()`.
3.  It then calls the final, generic implementation, `empty_generic`, passing the chosen allocator.

### Step 4: The Core Logic - `at::detail::empty_generic`

*   **File:** `aten/src/ATen/EmptyTensor.cpp`

This is where the magic happens. This function is backend-agnostic; it takes an allocator as an argument and doesn't care whether it's a CPU, CUDA, or any other allocator.

```cpp
// aten/src/ATen/EmptyTensor.cpp

TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  
  // 1. Calculate storage size
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());

  // 2. Allocate memory and create StorageImpl
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  // 3. Create the TensorImpl
  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
      
  // 4. Set tensor properties
  tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);

  if (memory_format_opt.has_value()) {
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  return tensor;
}
```

This process involves three key objects:

1.  **`Storage` and `StorageImpl`**: A `Storage` object represents a single, contiguous block of memory. The `StorageImpl` is the actual implementation that holds a `DataPtr`, which is a smart pointer to the raw memory allocated by the provided `allocator`. `empty_generic` calculates the required number of bytes and creates a `StorageImpl` of that size.

2.  **`Tensor` and `TensorImpl`**: The `at::Tensor` that users interact with is a lightweight handle (a smart pointer) to a `TensorImpl`. The `TensorImpl` is the object that truly represents the tensor. It holds a reference to the `StorageImpl` and contains metadata like sizes, strides, and data type. `empty_generic` creates a `TensorImpl` and links it to the newly created `StorageImpl`.

3.  **Setting Metadata**: Finally, the function sets the `sizes` and `strides` on the new `TensorImpl`. If a specific `memory_format` (like `ChannelsLast`) is requested, it adjusts the strides accordingly.

## 4. The CUDA Implementation of `at::empty`

The process for creating a CUDA tensor is remarkably similar, which is a testament to ATen's modular design.

The call stack is:
`at::empty` -> `at::native::empty_cuda` -> `at::detail::empty_cuda` -> `at::detail::empty_generic`

### Step 1 & 2: Public API and Native Kernel

*   **File:** `aten/src/ATen/native/cuda/TensorFactories.cu`

Just like the CPU case, a call to `at::empty` with a CUDA device option is dispatched to `at::native::empty_cuda`.

```cu
// aten/src/ATen/native/cuda/TensorFactories.cu

Tensor empty_cuda(IntArrayRef size, std::optional<ScalarType> dtype_opt, /*...other options...*/) {
  Tensor result = at::detail::empty_cuda(size, dtype_opt, /*...other options...*/);
  // ... (optional deterministic fill) ...
  return result;
}
```

### Step 3: The Detail Implementation - `at::detail::empty_cuda`

*   **File:** `aten/src/ATen/cuda/EmptyTensor.cpp`

This is where the primary difference lies. Instead of a CPU allocator, this function explicitly retrieves the CUDA allocator.

```cpp
// aten/src/ATen/cuda/EmptyTensor.cpp

TensorBase empty_cuda(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<Device> device_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
      
  // 1. Initialize CUDA and set the device
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  const auto device = device_or_default(device_opt);
  const DeviceGuard device_guard(device);
  
  // 2. Get the CUDA allocator
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  
  // 3. Call the generic implementation
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_generic(
      size, allocator, cuda_dks, dtype, memory_format_opt);
}
```
This function ensures the CUDA context is initialized for the correct device and then fetches the `CUDADeviceAllocator`. This allocator is responsible for managing memory on the GPU via `cudaMalloc` and `cudaFree`.

### Step 4: The Core Logic - `at::detail::empty_generic`

The final step is, once again, a call to the **exact same** `at::detail::empty_generic` function in `aten/src/ATen/EmptyTensor.cpp`. This function proceeds exactly as it did in the CPU case, but this time the `allocator` it receives is a `CUDADeviceAllocator`. When `storage_impl` is created, it calls the allocator's `allocate` method, which now results in memory being allocated on the GPU.

## 5. Summary: CPU vs. CUDA Implementation

| Feature               | CPU Implementation                                                                    | CUDA Implementation                                                                 |
| --------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Native Entrypoint** | `at::native::empty_cpu`                                                                 | `at::native::empty_cuda`                                                            |
| **File Location**     | `aten/src/ATen/native/TensorFactories.cpp`                                              | `aten/src/ATen/native/cuda/TensorFactories.cu`                                      |
| **Detail Function**   | `at::detail::empty_cpu`                                                                 | `at::detail::empty_cuda`                                                            |
| **File Location**     | `aten/src/ATen/EmptyTensor.cpp`                                                         | `aten/src/ATen/cuda/EmptyTensor.cpp`                                                |
| **Allocator**         | `c10::GetCPUAllocator()` or a pinned memory allocator.                                  | `at::cuda::getCUDADeviceAllocator()`                                                |
| **Core Logic**        | `at::detail::empty_generic` in `aten/src/ATen/EmptyTensor.cpp`                          | `at::detail::empty_generic` in `aten/src/ATen/EmptyTensor.cpp`                      |

## 6. Conclusion

The implementation of `at::empty` is a perfect example of the power of abstraction and modular design in PyTorch's backend. By separating the public API (`at::`), the native dispatch entry points (`at::native::`), and the low-level details (`at::detail::`), the system achieves several key goals:

*   **Code Reusability**: The core logic for creating a tensor's metadata (`TensorImpl`, `StorageImpl`) is shared across all backends in `empty_generic`.
*   **Extensibility**: Adding a new backend (e.g., for a new type of accelerator) is a matter of providing a new native kernel (e.g., `empty_my_backend`) and a corresponding allocator. The core logic remains unchanged.
*   **Clarity**: The separation of concerns makes the codebase easier to navigate. If you want to understand the specifics of CUDA allocation, you look at `at::detail::empty_cuda`. If you want to understand the generic process of creating a tensor from storage, you look at `at::detail::empty_generic`.

This layered and dispatched approach is fundamental to how PyTorch operates, allowing it to efficiently support a wide range of hardware and tensor types while maintaining a clean and consistent API.
