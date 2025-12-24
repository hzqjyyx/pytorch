# torch.max 调用流程详解

## 概述

本文档详细介绍了 `torch.max` 从 Python 用户API 到底层硬件执行的完整调用流程。`torch.max` 是 PyTorch 中重要的归约和比较操作，它支持三种不同的调用模式，根据参数自动选择相应的计算路径。

## 整体架构

```
用户层 (Python)
    ↓
API绑定层 (Python C Extension)
    ↓
分派层 (C++ Dispatcher)
    ↓
核心实现层 (ATen Native Functions)
    ↓
设备特定实现层 (CPU Kernels / CUDA Kernels)
    ↓
硬件执行层 (CPU / GPU)
```

## 1. Python 层入口

### 1.1 用户API

```python
# 模式1: 全局最大值 - 返回整个张量的最大值（标量）
result = torch.max(tensor)

# 模式2: 指定维度 - 返回指定维度的最大值和索引
values, indices = torch.max(tensor, dim=0)
values, indices = torch.max(tensor, dim=1, keepdim=True)

# 模式3: 逐元素比较 - 返回两个张量逐元素比较的最大值
result = torch.max(tensor1, tensor2)

# 输出参数版本
torch.max(tensor, out=result)
torch.max(tensor, dim=0, out=(values, indices))
```

### 1.2 API绑定

这些函数是对 C++ 扩展的直接绑定：`torch._C.max`

**文件位置**: `torch/__init__.py`

## 2. 核心定义层

### 2.1 native_functions.yaml 定义

**文件**: `aten/src/ATen/native/native_functions.yaml`

PyTorch 使用 YAML 文件定义所有原生操作的接口和分派规则：

```yaml
# 模式1: 全局最大值
- func: max(Tensor self) -> Tensor
  dispatch:
    CPU, CUDA: max
    MPS: max_mps
    QuantizedCPU: max_quantized_cpu

# 模式2: 指定维度（返回值和索引）
- func: max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  structured_delegate: max.dim_max
  dispatch:
    QuantizedCPU, QuantizedCUDA: qmax

# 模式2: 输出版本
- func: max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
  structured: True
  dispatch:
    CPU, CUDA: max_out
    MPS: max_out_mps

# 模式3: 二元比较（实际是 maximum 的别名）
- func: max.other(Tensor self, Tensor other) -> Tensor
  dispatch:
    CompositeExplicitAutograd: maximum

# 全局最大值输出版本
- func: max.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, CUDA: max_unary_out
    QuantizedCPU: max_quantized_unary_out
```

## 3. C++ 分派层

### 3.1 模式1: 全局最大值实现

**文件**: `aten/src/ATen/native/ReduceAllOps.cpp`

```cpp
Tensor max(const Tensor &self) {
  // 检查张量非空
  TORCH_CHECK(self.numel() > 0,
    "max(): Expected reduction dim to be specified for input.numel() == 0.");

  // 创建标量结果张量
  Tensor result = at::empty({}, self.options());

  // 分派到设备特定的实现
  max_all_stub(self.device().type(), result, self.contiguous());

  return result;
}

Tensor& max_unary_out(const Tensor &self, Tensor& out) {
  // 输出参数版本
  at::native::resize_output(out, {});
  max_all_stub(self.device().type(), out, self.contiguous());
  return out;
}
```

**主要功能**:
- 检查输入张量非空
- 创建标量输出张量
- 通过 `max_all_stub` 分派到 CPU/CUDA 实现
- 确保输入张量内存连续

### 3.2 模式2: 指定维度实现

**文件**: `aten/src/ATen/native/TensorCompare.cpp`

```cpp
TORCH_IMPL_FUNC(max_out)
(const Tensor& self,
 int64_t dim,
 bool keepdim,
 const Tensor& values,
 const Tensor& indices) {
  // 调用通用的 min/max 实现
  minmax_out_impl(self, dim, keepdim, values, indices, max_stub);
}
```

**minmax_out_impl 核心逻辑**:

```cpp
static void minmax_out_impl(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const Tensor& values,
    const Tensor& indices,
    const ReduceMinMaxStub& stub) {

  // 维度归一化
  dim = at::maybe_wrap_dim(dim, self.dim());

  // 检查维度大小非零
  TORCH_CHECK(self.size(dim) > 0,
    "max(): Expected reduction dim ", dim, " to have non-zero size.");

  // 分派到设备特定的 kernel
  stub(self.device().type(), values, indices, self, dim, keepdim);
}
```

### 3.3 模式3: 逐元素比较实现

**文件**: `aten/src/ATen/native/BinaryOps.cpp`

```cpp
// max 是 maximum 的别名
Tensor max(const Tensor& self, const Tensor& other) {
  return at::maximum(self, other);
}

Tensor& max_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::maximum_out(result, self, other);
}

// maximum 的实际实现
Tensor maximum(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::borrowing_binary_op(result, self, other);
  maximum_stub(iter.device_type(), iter);
  return iter.output();
}
```

**主要功能**:
- 使用 `TensorIterator` 处理广播和类型提升
- 通过 `maximum_stub` 分派到设备实现
- 支持任意形状的张量广播

## 4. CPU 实现层

### 4.1 全局最大值 CPU 实现

**文件**: `aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp`

```cpp
static void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    // Bool 类型: 任意 true 则结果为 true
    bool result_data = false;
    auto iter = TensorIteratorConfig()
      .add_const_input(input)
      .build();

    cpu_serial_kernel(iter, [&](const bool a) -> void {
      result_data = result_data || a;
    });
    result.fill_(result_data);

  } else if (input.scalar_type() == ScalarType::Long) {
    // int64_t 使用标量路径（避免向量化开销）
    reduce_all_impl<int64_t>(
      result, input,
      std::numeric_limits<int64_t>::lowest(),
      [=](int64_t a, int64_t b) -> int64_t {
        return max_impl(a, b);
      });

  } else {
    // 其他类型使用向量化实现
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(), "max_all", [&] {
        using Vec = Vectorized<at::opmath_type<scalar_t>>;

        reduce_all_impl_vec<scalar_t>(
          result, input,
          std::numeric_limits<scalar_t>::lowest(),
          // 标量归约函数
          [=](scalar_t a, scalar_t b) -> scalar_t {
            return max_impl(a, b);
          },
          // 向量归约函数
          [=](Vec a, Vec b) -> Vec {
            return maximum(a, b);
          });
    });
  }
}

// 注册分派
REGISTER_DISPATCH(max_all_stub, &max_all_kernel_impl)
```

**关键优化**:
- **向量化**: 使用 AVX/AVX2 指令并行处理多个元素
- **类型特化**: Bool 和 int64_t 使用特殊路径
- **NaN 处理**: `max_impl` 确保 NaN 传播

### 4.2 指定维度 CPU 实现

**文件**: `aten/src/ATen/native/cpu/TensorCompareKernel.cpp`

```cpp
static void max_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {

  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  AT_DISPATCH_ALL_TYPES_AND3(
    ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool,
    self.scalar_type(), "max_cpu", [&] {

      compare_base_kernel<scalar_t>(
        result, indice, self, dim, keepdim,
        [&](scalar_t* result_data,
            int64_t* indice_data,
            const scalar_t* self_data,
            auto self_dim_stride) {

          // 初始化为第一个元素
          scalar_t max_number = c10::load(self_data);
          int64_t index = 0;

          // 遍历指定维度
          for (const auto i : c10::irange(self_dim_size)) {
            scalar_t value = c10::load(&self_data[i * self_dim_stride]);

            // NaN 优先传播
            if (!(zabs_(value) <= zabs_(max_number))) {
              max_number = value;
              index = i;
              if (_isnan<scalar_t>(value)) {
                break;  // 遇到 NaN 立即停止
              }
            }
          }

          *result_data = max_number;
          *indice_data = index;
        }
      );
  });
}

// 注册分派
REGISTER_DISPATCH(max_stub, &max_kernel_impl)
```

**NaN 处理策略**:
- NaN 与任何值比较都返回 NaN
- 一旦遇到 NaN，立即返回该 NaN 及其索引
- 使用 `!(a <= b)` 而非 `a > b` 来正确处理 NaN

### 4.3 逐元素比较 CPU 实现

**文件**: `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`

```cpp
void maximum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // Bool: 逻辑或操作
    cpu_kernel(iter,
      [](bool a, bool b) -> bool {
        return a || b;
      });

  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf,
      iter.dtype(), "max_elementwise_cpu", [&]() {

        // 同时提供标量和向量化版本
        cpu_kernel_vec(
          iter,
          // 标量版本
          [](scalar_t a, scalar_t b) -> scalar_t {
            return max_impl(a, b);
          },
          // 向量化版本（AVX/AVX2）
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return maximum(a, b);
          });
    });
  }
}

REGISTER_DISPATCH(maximum_stub, &maximum_kernel)
```

**max_impl 实现**:

```cpp
template <typename scalar_t>
inline scalar_t max_impl(scalar_t a, scalar_t b) {
  // NaN 传播: 如果任一为 NaN，返回 NaN
  if (_isnan(a)) return a;
  if (_isnan(b)) return b;
  return a > b ? a : b;
}
```

## 5. CUDA 实现层

### 5.1 全局最大值 CUDA 实现

**文件**: `aten/src/ATen/native/cuda/ReduceOps.cpp`

```cpp
void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();

  // 创建归约迭代器
  auto iter = make_reduction(
    "max_all", result, input,
    IntArrayRef{},  // 空维度表示全局归约
    false,          // keepdim
    dtype);

  // 启动 CUDA kernel
  max_all_launch_kernel(iter);
}

REGISTER_CUDA_DISPATCH(max_all_stub, &max_all_kernel_impl)
```

**CUDA Kernel 启动**: `aten/src/ATen/native/cuda/ReduceMaxValuesKernel.cu`

```cpp
void max_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
    kBFloat16, kHalf, kBool,
    iter.input_dtype(), "max_all_cuda", [&] {
      max_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

template <typename scalar_t, typename acc_t = scalar_t>
void max_values_kernel_cuda_impl(TensorIterator& iter) {
  // 使用 GPU 归约框架
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter,
    func_wrapper<acc_t>(MaxNanFunctor<acc_t>()),
    at::numeric_limits<acc_t>::lower_bound());  // 初始值
}

// Max 归约函数（NaN 优先）
template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    // NaN 传播: NaN 或 a > b 时返回 a
    return (at::_isnan(a) || a > b) ? a : b;
  }
};
```

**GPU 归约策略**:
- **两阶段归约**: 块内归约 + 块间归约
- **共享内存**: 使用 shared memory 加速块内归约
- **Warp shuffle**: 利用 warp 级原语优化

### 5.2 指定维度 CUDA 实现

**文件**: `aten/src/ATen/native/cuda/ReduceOps.cpp`

```cpp
void max_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {

  // 创建归约迭代器（同时输出值和索引）
  auto iter = meta::make_reduction(
    self, result, indice, dim, keepdim,
    self.scalar_type(),  // 值类型
    kLong);              // 索引类型

  max_launch_kernel(iter);
}

REGISTER_CUDA_DISPATCH(max_stub, &max_kernel_impl)
```

**CUDA Kernel**: `aten/src/ATen/native/cuda/ReduceMaxValuesKernel.cu`

```cpp
void max_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
    kBFloat16, kHalf, kBool,
    iter.input_dtype(), "max_cuda", [&]() {

      // 使用 pair 同时归约值和索引
      gpu_reduce_kernel<scalar_t, scalar_t>(
        iter,
        MaxOps<scalar_t>{},
        thrust::pair<scalar_t, int64_t>(
          at::numeric_limits<scalar_t>::lower_bound(), 0));
  });
}
```

**MaxOps 结构**: `aten/src/ATen/native/SharedReduceOps.h`

```cpp
template <typename scalar_t>
struct MaxOps {
  using arg_t = thrust::pair<scalar_t, int64_t>;

  // 归约单个元素
  __device__ __forceinline__ arg_t reduce(
      arg_t acc, scalar_t data, int64_t idx) const {
    // NaN 优先，否则比较值
    if (_isnan(data) || data > acc.first) {
      return arg_t{data, idx};
    }
    return acc;
  }

  // 合并两个归约结果
  __device__ __forceinline__ arg_t combine(arg_t a, arg_t b) const {
    if (_isnan(a.first) || a.first > b.first) {
      return a;
    }
    return b;
  }

  // 提取最终值
  __device__ __forceinline__ scalar_t project(arg_t arg) const {
    return arg.first;
  }
};
```

### 5.3 逐元素比较 CUDA 实现

**文件**: `aten/src/ATen/native/cuda/MaxMinElementwiseKernel.cu`

```cpp
void maximum_kernel_cuda(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // Bool: 逻辑或
    opmath_symmetric_gpu_kernel_with_scalars<bool>(
      iter, []GPU_LAMBDA(bool a, bool b) -> bool {
        return a || b;
      });

  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    // 整数类型
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return ::max(a, b);
        });
    });

  } else {
    // 浮点类型（需要 NaN 处理）
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(), "max_elementwise_cuda", [&]() {

        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
            // NaN 传播
            if (a != a) return a;  // a 是 NaN
            else if (b != b) return b;  // b 是 NaN
            else return ::max(a, b);
          });
    });
  }
}

REGISTER_DISPATCH(maximum_stub, &maximum_kernel_cuda)
```

## 6. 分派机制

### 6.1 Stub 声明

**文件**: `aten/src/ATen/native/ReduceAllOps.h`

```cpp
using reduce_all_fn = void (*)(Tensor& result, const Tensor& self);

DECLARE_DISPATCH(reduce_all_fn, max_all_stub);
```

**文件**: `aten/src/ATen/native/TensorCompare.h`

```cpp
using structured_reduce_minmax_fn = void (*)(
  const Tensor& values,
  const Tensor& indices,
  const Tensor& self,
  int64_t dim,
  bool keepdim);

DECLARE_DISPATCH(structured_reduce_minmax_fn, max_stub);
```

**文件**: `aten/src/ATen/native/BinaryOps.h`

```cpp
using binary_fn = void (*)(TensorIteratorBase&);

DECLARE_DISPATCH(binary_fn, maximum_stub);
```

### 6.2 分派流程

```cpp
// 1. 用户调用
torch.max(tensor)

// 2. C++ 入口
at::max(tensor)

// 3. 分派到设备
max_all_stub(tensor.device().type(), result, tensor)

// 4. 设备实现
// CPU: max_all_kernel_impl (ReduceAllOpsKernel.cpp)
// CUDA: max_all_kernel_impl (ReduceOps.cpp)

// 5. 硬件执行
// CPU: AVX/AVX2 向量化指令
// CUDA: GPU 并行归约
```

## 7. 性能优化特性

### 7.1 向量化优化

**CPU 向量化**:
```cpp
// 使用 AVX/AVX2 一次处理多个元素
using Vec = Vectorized<float>;  // 通常是 8 个 float

Vec a_vec = Vec::loadu(a_ptr);
Vec b_vec = Vec::loadu(b_ptr);
Vec max_vec = maximum(a_vec, b_vec);  // 并行比较
max_vec.store(result_ptr);
```

**CUDA 并行化**:
```cpp
// 每个线程处理一个元素
__global__ void maximum_kernel(float* out, float* a, float* b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = fmaxf(a[idx], b[idx]);
  }
}
```

### 7.2 归约优化

**两阶段归约**:
```
输入: [N 个元素]
    ↓
阶段1: 块内归约（共享内存）
    [Block 0] → max_0
    [Block 1] → max_1
    ...
    [Block K] → max_K
    ↓
阶段2: 块间归约
    max(max_0, max_1, ..., max_K) → 最终结果
```

**Warp Shuffle 优化**:
```cpp
// 使用 warp 级原语避免共享内存
__device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}
```

### 7.3 内存访问优化

- **合并访问**: 确保线程访问连续内存
- **共享内存**: 减少全局内存访问
- **寄存器优化**: 最大化寄存器使用

### 7.4 NaN 处理

所有实现都遵循 IEEE 754 标准:
- NaN 与任何值比较都返回 NaN
- NaN 在归约中优先传播
- 使用 `!(a <= b)` 而非 `a > b` 来正确处理 NaN

## 8. 完整调用流程图

```
torch.max(tensor)
    ↓
torch._C.max                              # Python C扩展绑定
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│模式1: 全局最大值  │模式2: 指定维度   │模式3: 逐元素比较 │
│at::max(self)    │at::max(self,dim)│at::max(self,other)│
│↓                │↓                │↓                │
│max_all_stub     │max_stub         │maximum_stub     │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                   ↓                   ↓
┌─────────────────┬─────────────────┬─────────────────┐
│CPU 实现          │CPU 实现          │CPU 实现          │
│• 向量化归约      │• 维度遍历        │• 向量化比较      │
│• AVX/AVX2       │• NaN 优先        │• 广播支持        │
│• 并行化          │• 索引追踪        │• TensorIterator │
└─────────────────┴─────────────────┴─────────────────┘
┌─────────────────┬─────────────────┬─────────────────┐
│CUDA 实现         │CUDA 实现         │CUDA 实现         │
│• 两阶段归约      │• 块内+块间归约   │• 逐元素 kernel   │
│• 共享内存        │• 值+索引 pair    │• 合并访问        │
│• Warp shuffle   │• NaN 传播        │• NaN 处理        │
└─────────────────┴─────────────────┴─────────────────┘
    ↓
硬件执行 (CPU cores / GPU SMs)
```

## 9. 调试和优化建议

### 9.1 性能分析

```python
# 检查张量连续性
if not tensor.is_contiguous():
    tensor = tensor.contiguous()

# 使用 profiler 分析性能
with torch.profiler.profile() as prof:
    result = torch.max(tensor)
print(prof.key_averages().table())

# CUDA 性能优化
torch.backends.cudnn.benchmark = True
```

### 9.2 常见性能瓶颈

1. **内存不连续**: 使用 `tensor.contiguous()` 确保连续性
2. **小张量开销**: 小张量的 kernel 启动开销可能超过计算时间
3. **频繁的 CPU-GPU 传输**: 尽量在同一设备上完成所有操作
4. **不必要的同步**: 避免频繁调用 `.item()` 或 `.cpu()`

### 9.3 最佳实践

- **批量操作**: 尽可能使用批量操作而非循环
- **原地操作**: 使用 `out` 参数避免内存分配
- **维度选择**: 选择合适的归约维度以优化内存访问模式
- **数据类型**: 使用适当的数据类型（float16 可能更快但精度较低）

## 10. 相关函数对比

| 函数 | 用途 | 返回值 | NaN 处理 | 维度支持 |
|------|------|--------|----------|----------|
| `torch.max` | 最大值 | 值或(值,索引) | NaN 优先 | 全局/指定维度 |
| `torch.maximum` | 逐元素最大 | 张量 | NaN 传播 | 广播支持 |
| `torch.amax` | 最大值（仅值） | 值 | NaN 优先 | 多维度 |
| `torch.argmax` | 最大值索引 | 索引 | NaN 优先 | 全局/指定维度 |
| `torch.topk` | Top-K 最大值 | (值,索引) | NaN 优先 | 指定维度 |

## 11. 总结

`torch.max` 是 PyTorch 中设计精巧的归约和比较操作，它通过以下机制提供高效的实现：

- **智能分派**: 根据参数自动选择三种不同的计算模式
- **设备优化**: CPU 使用向量化指令，CUDA 使用并行归约
- **NaN 处理**: 严格遵循 IEEE 754 标准，确保数值稳定性
- **性能优化**: 两阶段归约、共享内存、warp shuffle 等优化技术

理解 `torch.max` 的调用流程有助于：
- **性能优化**: 选择合适的调用模式和参数
- **调试定位**: 理解性能瓶颈的根源
- **扩展开发**: 为自定义归约操作提供参考

无论是简单的全局最大值还是复杂的多维归约，`torch.max` 都能自动选择最优的执行路径，体现了 PyTorch 作为现代深度学习框架的设计哲学：简单易用的 API 背后是复杂而精巧的优化机制。
