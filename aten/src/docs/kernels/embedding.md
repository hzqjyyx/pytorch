# A Deep Dive into the `embedding` Kernel (Forward Pass)

This document provides a detailed, educative walkthrough of how the `torch.nn.Embedding` forward pass is implemented in PyTorch. We will trace the execution from the high-level Python API down to the device-specific C++ kernels for both CPU and CUDA, focusing on how the data lookup is performed.

This explanation is based on the PyTorch codebase as of September 2025.

## 1. Introduction: `embedding` as a Lookup Operation

The `embedding` operation is fundamentally a lookup table. Given a tensor of indices (usually `LongTensor`), it retrieves the corresponding vectors from a weight matrix (the "embedding" matrix). Unlike a "view" operation like `transpose`, an embedding is a **data-copying operation**. It creates a new output tensor and fills it with data copied from the weight matrix.

A typical call in Python looks like this:

```python
# Create an embedding layer for 10 items with 3-dimensional vectors
embedding_layer = torch.nn.Embedding(num_embeddings=10, embedding_dim=3)

# Create a batch of indices to look up
input_indices = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

# Perform the lookup
output = embedding_layer(input_indices)
```

This simple call initiates a chain of events that culminates in a highly optimized, device-specific kernel execution.

## 2. The Python Call Stack

The journey begins in the user-facing Python API and descends through several layers before reaching the C++ backend.

### Step 1: The `nn.Module`
- **File**: `torch/nn/modules/sparse.py`
- **Line**: `15` (approx.)
- **Code**: `class Embedding(Module):`
- **Explanation**: The `torch.nn.Embedding` class is a stateful `nn.Module`. Its primary role is to initialize and hold the `weight` parameter, which is the actual embedding matrix. When the module instance is called (e.g., `embedding_layer(...)`), it executes its `forward` method.

### Step 2: The Functional API
- **File**: `torch/nn/functional.py`
- **Line**: `2551` (approx.)
- **Code**: `return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)`
- **Explanation**: The `forward` method of the `Embedding` module is a wrapper around the stateless `torch.nn.functional.embedding` function. This is a common PyTorch pattern that separates state management (`nn.Module`) from the core logic (`nn.functional`). This function, after some validation, makes the crucial call that crosses the boundary into the C++ world.

### Step 3: The C++ Boundary
- **Code**: `torch.embedding(...)`
- **Explanation**: This call invokes a C++ function that has been exposed to Python via the pybind11 library. This is the entry point into PyTorch's ATen library.

## 3. The C++ Dispatch Mechanism

Once in C++, the call doesn't go to a single function but to the **ATen dispatcher**.

- **File**: `aten/src/ATen/native/native_functions.yaml`
- **Line**: `2292` (approx.)
- **Code**: `- func: embedding(Tensor weight, Tensor indices, ...)`
- **Explanation**: This YAML file declaratively defines the signature for all native ATen functions. The dispatcher reads this information and, based on the `Device` of the input tensors (`weight` and `indices`), routes the call to the correct registered kernel. If the tensors are on a CPU, it calls the CPU kernel; if they are on a CUDA device, it calls the CUDA kernel.

For the forward pass, the `embedding` function is essentially a specialized form of another core ATen operator: `index_select`. The following sections will therefore analyze the implementation of `index_select`, which performs the core logic.

## 4. The CPU Implementation

When the dispatcher selects the CPU backend, it ultimately calls a kernel optimized for CPU execution.

- **File**: `aten/src/ATen/native/cpu/Indexing.cpp`

The core logic for `index_select` on the CPU is found in the `index_select_kernel_impl` function.

```cpp
// In aten/src/ATen/native/cpu/Indexing.cpp

static void index_select_kernel_impl(
    char* C_data, const int64_t C_stride,
    const char* A_data, const int64_t A_stride,
    const int64_t* B_data, const int64_t B_size,
    const int64_t slice_size) {
  at::parallel_for(0, B_size, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      const int64_t index = B_data[i];
      // C_data is the output tensor data
      // A_data is the weight matrix data
      char* C_ptr = C_data + i * C_stride;
      const char* A_ptr = A_data + index * A_stride;
      memcpy(C_ptr, A_ptr, slice_size);
    }
  });
}
```

**Explanation:**
1.  **`parallel_for`**: The operation is parallelized across the number of indices to look up. `at::parallel_for` uses a thread pool (e.g., OpenMP) to split the work among multiple CPU cores.
2.  **Iteration**: Each thread iterates through a subset of the `indices` tensor (represented by `B_data`).
3.  **Address Calculation**: For each index, it calculates the source pointer (`A_ptr`) in the weight matrix and the destination pointer (`C_ptr`) in the output tensor. This is done using the tensor's strides.
4.  **`memcpy`**: The `memcpy` function performs the actual data copy, moving the bytes for one embedding vector from the weight matrix to the output tensor.

## 5. The CUDA Implementation

When tensors are on a GPU, the dispatcher calls a CUDA kernel, which is a function designed for massive parallel execution.

- **File**: `aten/src/ATen/native/cuda/Indexing.cu`

The CUDA implementation of `index_select` uses the `index_select_kernel` function.

```cu
// In aten/src/ATen/native/cuda/Indexing.cu

template <typename scalar_t, typename index_t>
__global__ void index_select_kernel(
    TensorInfo<scalar_t, index_t> dst,
    TensorInfo<scalar_t, index_t> src,
    TensorInfo<index_t, index_t> indices,
    int64_t inner_size) {

  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < indices.sizes[0];
       i += blockDim.x * gridDim.x) {

    // Each thread gets an index from the indices tensor
    index_t index = indices.data[i];
    
    // Get pointers to the source (weight) and destination (output)
    scalar_t* dst_ptr = dst.data + i * dst.strides[0];
    scalar_t* src_ptr = src.data + index * src.strides[0];

    // Copy the embedding vector element by element
    for (int64_t j = 0; j < inner_size; ++j) {
      dst_ptr[j] = src_ptr[j];
    }
  }
}
```

**Explanation:**
1.  **`__global__`**: This keyword marks `index_select_kernel` as a function that runs on the GPU and can be called from the CPU.
2.  **Grid-Stride Loop**: The kernel is launched with a grid of thread blocks. The `for` loop is a standard "grid-stride loop" pattern in CUDA, which allows a fixed number of threads to process an arbitrarily large number of indices.
3.  **Parallel Lookup**: Each GPU thread independently reads an index from the `indices` tensor.
4.  **Parallel Copy**: Each thread copies one entire embedding vector from the source (the `weight` tensor in global GPU memory) to the destination (the output tensor, also in global GPU memory). Because thousands of threads execute this simultaneously, the overall operation is extremely fast.

## 6. Summary: CPU vs. CUDA Forward Pass

| Feature | CPU Implementation (`index_select_kernel_impl`) | CUDA Implementation (`index_select_kernel`) |
|---|---|---|
| **File** | `aten/src/ATen/native/cpu/Indexing.cpp` | `aten/src/ATen/native/cuda/Indexing.cu` |
| **Execution Model** | Parallelism via multi-core threading (`at::parallel_for`). | Massive parallelism via thousands of GPU threads. |
| **Core Operation** | `memcpy` to copy entire vectors in a loop. | Per-thread loop to copy vector elements. |
| **Granularity** | A CPU thread processes a chunk of indices. | A GPU thread typically processes one index at a time. |

## 7. Conclusion

The `embedding` forward pass is a clear illustration of PyTorch's core design philosophy. A simple Python API call is translated through layers of abstraction into a device-specific, highly optimized kernel. The dispatcher cleanly separates the generic logic from the hardware-specific implementations, allowing the same user code to run efficiently on different backends. The use of `index_select` as the underlying mechanism demonstrates how complex operations are often composed from a set of powerful, fundamental primitives within the ATen library.
