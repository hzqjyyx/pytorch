# 如何找到算子的实现代码

## 1. 概述

给定 `native_functions.yaml` 中的算子定义，本文档说明如何找到它的具体 C++ 实现代码。

---

## 2. native_functions.yaml 中的 dispatch 字段

### 2.1 基本结构

```yaml
- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: mul_out          # 通用后端
    MPS: mul_out_mps            # Apple Metal
    SparseCPU: mul_out_sparse_cpu
    SparseCUDA: mul_out_sparse_cuda
    MkldnnCPU: mkldnn_mul_out   # Intel MKL-DNN
```

### 2.2 dispatch 字段的作用

- **后端标签**：CPU, CUDA, SparseCPU, SparseCUDA, MkldnnCPU, MPS, ZeroTensor, NestedTensorCPU 等
- **函数映射**：每个后端对应一个 C++ 函数名
- **默认分发**：如果不指定 dispatch，默认为 `CompositeImplicitAutograd`

---

## 3. 文件位置规律

### 3.1 目录结构

```
aten/src/ATen/native/
├── BinaryOps.cpp                    # 二元算子通用实现
├── UnaryOps.cpp                     # 一元算子通用实现
├── Activation.cpp                   # 激活函数
├── LinearAlgebra.cpp                # 线性代数
├── cpu/
│   ├── BinaryOpsKernel.cpp          # CPU 二元算子内核
│   └── UnaryOpsKernel.cpp           # CPU 一元算子内核
├── cuda/
│   ├── BinaryMulKernel.cu           # CUDA mul 实现
│   ├── BinaryMiscOpsKernels.cu      # 其他二元算子
│   └── Blas.cpp                     # BLAS 操作
├── mkldnn/
│   └── BinaryOps.cpp                # MKL-DNN 优化实现
├── sparse/
│   └── SparseTensorMath.cpp         # 稀疏张量实现
└── mps/
    └── operations/                   # Apple Metal 实现
```

### 3.2 命名对应规则

| 后端 | 文件位置 | 函数命名 |
|------|---------|---------|
| **通用** | `native/{Op}Ops.cpp` | `TORCH_IMPL_FUNC(xxx_out)` |
| **CPU** | `native/cpu/{Op}Kernel.cpp` | `xxx_kernel` |
| **CUDA** | `native/cuda/{Op}Kernel.cu` | `xxx_kernel_cuda` |
| **MKL-DNN** | `native/mkldnn/{Op}.cpp` | `mkldnn_xxx` |
| **Sparse** | `native/sparse/Sparse{Op}.cpp` | `xxx_sparse` |
| **MPS** | `native/mps/operations/` | `xxx_out_mps` |

---

## 4. 完整追踪示例：mul 算子

### Step 1: 找到 YAML 定义

**位置**: `native_functions.yaml:4243-4254`

```yaml
- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: mul_out
    MPS: mul_out_mps
    SparseCPU: mul_out_sparse_cpu
    SparseCUDA: mul_out_sparse_cuda
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: mul_out_sparse_csr
    MkldnnCPU: mkldnn_mul_out
  tags: pointwise
```

### Step 2: 找到通用实现

**位置**: `aten/src/ATen/native/BinaryOps.cpp:440-444`

```cpp
TORCH_IMPL_FUNC(mul_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  mul_stub(device_type(), *this);
}
```

**关键点**：
- `TORCH_IMPL_FUNC` 是 structured kernel 的实现宏
- 调用 `mul_stub` 进行设备相关的分发
- `device_type()` 返回当前设备类型
- `*this` 是已构建好的 TensorIterator

### Step 3: 找到 Stub 声明

**位置**: `aten/src/ATen/native/BinaryOps.h:60`

```cpp
DECLARE_DISPATCH(structured_binary_fn, mul_stub)
```

函数指针类型定义：
```cpp
using structured_binary_fn = void(*)(TensorIteratorBase&);
```

### Step 4: 找到 CPU 实现

**位置**: `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:126-175`

```cpp
void mul_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (dtype == ScalarType::Bool) {
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } else if (dtype == kComplexHalf) {
    cpu_kernel(iter, ...);  // ComplexHalf 特殊处理
  } else if (iter.is_scalar(2) && ...) {
    // 标量优化路径
    AT_DISPATCH_REDUCED_FLOATING_TYPES(...);
  } else {
    // 通用路径
  }
}

// 注册到 stub（第 1373 行）
REGISTER_DISPATCH(mul_stub, &mul_kernel)
```

### Step 5: 找到 CUDA 实现

**位置**: `aten/src/ATen/native/cuda/BinaryMulKernel.cu:1-49`

```cpp
void mul_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
#if AT_USE_JITERATOR()
    static const auto mul_string = jiterator_stringify(
        template <typename T> T mul_kernel(T a, T b) { return a * b; });
    opmath_jitted_gpu_kernel_with_scalars<mul_name, scalar_t, scalar_t>(
        iter, mul_string);
#else
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, binary_internal::MulFunctor<opmath_t>());
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(...) {
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, binary_internal::MulFunctor<opmath_t>());
    }
  }
}

// 注册到 stub（第 46 行）
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda)
```

---

## 5. Stub 机制详解

### 5.1 三个关键宏

```cpp
// 1. 声明 stub（在 .h 文件中）
DECLARE_DISPATCH(structured_binary_fn, mul_stub);

// 2. 定义 stub（在 .cpp 文件中）
DEFINE_DISPATCH(mul_stub);

// 3. 注册实现（在各后端的 Kernel 文件中）
REGISTER_DISPATCH(mul_stub, &mul_kernel);      // CPU
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda); // CUDA
```

### 5.2 工作原理

```cpp
// 调用时
mul_stub(device_type(), *this);

// 等价于
if (device_type() == CPU) {
    mul_kernel(*this);
} else if (device_type() == CUDA) {
    mul_kernel_cuda(*this);
}
```

Stub 本质上是一个函数指针表，在编译时由各后端的 `REGISTER_DISPATCH` 填充。

---

## 6. 查找步骤总结

```
Step 1: 在 native_functions.yaml 中搜索 "func: {op_name}"
        → 找到算子定义和 dispatch 表

Step 2: 如果有 structured_delegate
        → 追踪到 .out 变体

Step 3: 查看 dispatch 字段
        → 记录各后端的函数名

Step 4: 搜索函数名定位实现
        - 通用：grep "TORCH_IMPL_FUNC({func_name})" aten/src/ATen/native/
        - CPU：  grep "{func_name}" aten/src/ATen/native/cpu/
        - CUDA： grep "{func_name}" aten/src/ATen/native/cuda/
```

---

## 7. 常用搜索命令

```bash
# 找某个算子的所有 YAML 定义
grep -n "func: add" aten/src/ATen/native/native_functions.yaml

# 找通用实现（TORCH_IMPL_FUNC）
grep -rn "TORCH_IMPL_FUNC(add_out)" aten/src/ATen/native/

# 找 stub 声明
grep -rn "DECLARE_DISPATCH.*add_stub" aten/src/ATen/native/

# 找 stub 注册
grep -rn "REGISTER_DISPATCH(add_stub" aten/src/ATen/native/

# 找 CPU 内核实现
grep -rn "add_kernel" aten/src/ATen/native/cpu/

# 找 CUDA 内核实现
grep -rn "add_kernel_cuda" aten/src/ATen/native/cuda/
```

---

## 8. 特殊情况

### 8.1 没有 dispatch 字段

如果 YAML 中没有 dispatch 字段，则使用默认的 Composite 实现：

```yaml
- func: broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)
  variants: function, method
  # 没有 dispatch 字段 → CompositeImplicitAutograd
```

实现通常在同名的 .cpp 文件中：
```cpp
Tensor broadcast_to_symint(const Tensor& self, SymIntArrayRef size) {
  return self.expand_symint(size);
}
```

### 8.2 特殊后端的独立实现

某些后端（Sparse、MKL-DNN 等）有完全独立的实现，不走 stub 机制：

```yaml
dispatch:
  SparseCPU, SparseCUDA: add_sparse  # 独立实现
```

这些函数直接定义在对应的文件中：
```cpp
// aten/src/ATen/native/sparse/SparseTensorMath.cpp
Tensor add_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 完整的稀疏张量加法实现
}
```

---

## 9. 关键文件索引

| 用途 | 路径 |
|------|------|
| 所有算子定义 | `aten/src/ATen/native/native_functions.yaml` |
| 二元算子通用 | `aten/src/ATen/native/BinaryOps.cpp` |
| 二元算子 stub 声明 | `aten/src/ATen/native/BinaryOps.h` |
| CPU 二元算子内核 | `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` |
| CUDA mul 内核 | `aten/src/ATen/native/cuda/BinaryMulKernel.cu` |
| CUDA 其他二元算子 | `aten/src/ATen/native/cuda/BinaryMiscOpsKernels.cu` |
| MKL-DNN 实现 | `aten/src/ATen/native/mkldnn/BinaryOps.cpp` |
| Sparse 实现 | `aten/src/ATen/native/sparse/SparseTensorMath.cpp` |
| 开发指南 | `aten/src/ATen/native/README.md` |
