# Composite Ops 与 Meta/Impl 分离

## 1. Composite Ops

### 1.1 什么是 Composite Op

Composite op（组合算子）不是独立实现的，而是通过调用其他基础算子组合而成。

```yaml
- func: broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)
  dispatch:
    CompositeImplicitAutograd: broadcast_to_symint
```

```cpp
// 实现只是调用另一个算子
Tensor broadcast_to_symint(const Tensor& self, SymIntArrayRef size) {
  return self.expand_symint(size);
}
```

### 1.2 两种 Composite 类型

| 类型 | 含义 | Autograd 处理 |
|------|------|--------------|
| `CompositeExplicitAutograd` | 显式定义梯度规则 | 手动编写 backward |
| `CompositeImplicitAutograd` | 自动推导梯度 | 通过 tracing 组件算子推导 |

---

## 2. CompositeExplicitAutograd 示例

### 2.1 add.Scalar

**位置**: `native_functions.yaml:620-625`

```yaml
- func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: add
  tags: [core, pointwise]
```

**实现位置**: `aten/src/ATen/native/BinaryOps.cpp:1159-1165`

```cpp
Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  // 将 Scalar 包装成 Tensor，然后调用 Tensor 版本
  return at::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.add_(wrapped_scalar_tensor(other), alpha);
}
```

**关键点**：`add.Scalar` 只是将 Scalar 包装成 Tensor，然后委托给 `add.Tensor`。

### 2.2 _is_all_true / _is_any_true

**位置**: `native_functions.yaml:680-688`

```yaml
- func: _is_all_true(Tensor self) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: _is_all_true

- func: _is_any_true(Tensor self) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: _is_any_true
```

**实现位置**: `aten/src/ATen/native/ReduceOps.cpp:2187-2194`

```cpp
Tensor _is_all_true(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.scalar_type() == at::kBool);
  return self.all();  // 调用 all() 算子
}

Tensor _is_any_true(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.scalar_type() == at::kBool);
  return self.any();  // 调用 any() 算子
}
```

---

## 3. CompositeImplicitAutograd 示例

### 3.1 broadcast_to

**位置**: `native_functions.yaml:1382-1386`

```yaml
- func: broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)
  variants: function, method
  dispatch:
    CompositeImplicitAutograd: broadcast_to_symint
```

**实现位置**: `aten/src/ATen/native/TensorShape.cpp:644-646`

```cpp
Tensor broadcast_to_symint(const Tensor& self, SymIntArrayRef size) {
  return self.expand_symint(size);  // 只是调用 expand
}
```

### 3.2 chunk

**位置**: `native_functions.yaml:1484-1485`

```yaml
- func: chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
  dispatch:
    CompositeImplicitAutograd: chunk
```

**实现位置**: `aten/src/ATen/native/TensorShape.cpp:1069-1089`

```cpp
std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  const auto dim_size = self.sym_size(dim);
  auto split_size = (dim_size + chunks - 1) / chunks;

  if (split_size == 0 && dim_size == 0) {
    std::vector<c10::SymInt> split_sizes(chunks, split_size);
    return self.split_with_sizes_symint(split_sizes, dim);
  } else {
    return self.split_symint(std::move(split_size), dim);
  }
}
```

**关键点**：`chunk` 通过调用 `split` 或 `split_with_sizes` 实现。

---

## 4. Fallback 机制

### 4.1 Fallback 的意义

当某个 dispatch key（如 CUDA）没有特定实现时，系统会 fallback 到 Composite 实现。

```
调用 my_op(tensor)
    ↓
检查 CUDA 实现 → 没有
    ↓
检查 CompositeExplicitAutograd → 有
    ↓
执行 Composite 实现（调用其他基础算子）
    ↓
这些基础算子各自 dispatch 到 CUDA
```

### 4.2 Fallback 相关文件

| 文件 | 用途 |
|------|------|
| `aten/src/ATen/FunctionalizeFallbackKernel.cpp` | Functionalization fallback |
| `aten/src/ATen/native/CPUFallback.cpp` | CPU fallback |
| `aten/src/ATen/native/NegateFallback.cpp` | Conjugate 相关 fallback |

---

## 5. Meta/Impl 分离

### 5.1 核心概念

Structured kernel 将算子分为两部分：

| 部分 | 职责 | 运行位置 |
|------|------|---------|
| **Meta** | 计算输出 shape、dtype、strides | 任何 device |
| **Impl** | 执行实际计算 | 特定 device |

### 5.2 为什么要分离

1. **符号执行**：可以只跑 Meta 来推导 shape，不分配内存
2. **代码复用**：Meta 在所有后端共享
3. **一致性**：所有后端的 shape 检查逻辑相同
4. **编译优化**：可以在编译期进行 shape 验证

---

## 6. Meta/Impl 示例

### 6.1 threshold 算子

**YAML 定义**: `native_functions.yaml:6068-6088`

```yaml
- func: threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
  variants: function
  structured_delegate: threshold.out

- func: threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
  variants: method
  structured_delegate: threshold.out

- func: threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: threshold_out
    MPS: threshold_out_mps
```

**Meta 函数**: `aten/src/ATen/native/Activation.cpp:86-97`

```cpp
namespace at::meta {

TORCH_META_FUNC(threshold)(const Tensor& self, const Scalar& threshold,
                           const Scalar& value) {
  const Tensor& result = maybe_get_output();
  build(TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .add_output(result)
    .add_const_input(self)
    .add_const_input(self)  // other
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true));
}

}  // namespace at::meta
```

**Impl 函数**: `Activation.cpp:684-688`

```cpp
TORCH_IMPL_FUNC(threshold_out)(const Tensor& self, const Scalar& threshold,
                               const Scalar& value, const Tensor& result) {
  threshold_stub(device_type(), *this, threshold, value);
}
```

### 6.2 elu 算子

**Meta 函数**: `Activation.cpp:113-117`

```cpp
TORCH_META_FUNC(elu) (
  const Tensor& self, const Scalar& alpha,
  const Scalar& scale, const Scalar& input_scale
) {
  build_unary_op(maybe_get_output(), self);
}
```

**Impl 函数**: `Activation.cpp:270-274`

```cpp
TORCH_IMPL_FUNC(elu_out) (
  const Tensor& self, const Scalar& alpha,
  const Scalar& scale, const Scalar& input_scale,
  const Tensor& result
) {
  elu_stub(device_type(), *this, alpha, scale, input_scale);
}
```

### 6.3 Binary Operations

**Meta 函数**: `BinaryOps.cpp:151-156`

```cpp
namespace at::meta {

TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

}
```

**常用 build 函数**：

| 函数 | 用途 |
|------|------|
| `build_unary_op()` | 一元操作（如 threshold、elu） |
| `build_borrowing_binary_op()` | 二元操作（如 add、mul、sub） |
| `build_borrowing_binary_float_op()` | 二元浮点操作（如 div） |

---

## 7. Impl 函数的通用模式

**位置**: `BinaryOps.cpp:433-460`

```cpp
TORCH_IMPL_FUNC(sub_out) (
  const Tensor& self, const Tensor& other,
  const Scalar& alpha, const Tensor& result
) {
  add_stub(device_type(), *this, -alpha);  // sub 复用 add
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}

TORCH_IMPL_FUNC(mul_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  mul_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(div_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  div_true_stub(device_type(), *this);
}
```

**关键特点**：
- 都接收 `device_type()` 进行二级 dispatch
- 调用对应的 stub（如 `add_stub`, `mul_stub`）
- `*this` 是已构建好的 TensorIterator 对象

---

## 8. 调用流程

```
用户调用: torch.nn.functional.elu(input, alpha=1.0)
    ↓
Dispatch 阶段:
    - 查找 elu.out 的实现
    - 根据 device (CPU/CUDA/MPS) 选择实现
    ↓
Meta 函数阶段:
    - 创建输出张量 result
    - 通过 build_unary_op 设置 shape/dtype
    - 构建 TensorIterator
    ↓
Impl 函数阶段:
    - 接收 TensorIterator
    - 调用 elu_stub
    - 执行实际计算
    ↓
返回结果
```

---

## 9. Meta 函数的独立运行

Meta 函数可以独立运行，用于 shape 推导而不执行实际计算：

```python
# Python 中使用 Meta tensor
meta_input = torch.randn(3, 4, device='meta')
meta_output = torch.elu(meta_input)
print(meta_output.shape)  # torch.Size([3, 4])
print(meta_output.device)  # meta
# 不分配实际内存，只计算 shape
```

---

## 10. 总结对比

| 特性 | CompositeExplicitAutograd | CompositeImplicitAutograd | Structured Kernel |
|------|---------------------------|---------------------------|-------------------|
| 实现方式 | 调用其他算子组合 | 调用其他算子组合 | Meta + Impl 分离 |
| Autograd | 显式定义 | 隐式自动推导 | 自动生成 |
| Device 支持 | 通过组件算子 | 通过组件算子 | Meta 共享，Impl 多份 |
| 例子 | add.Scalar, _is_all_true | broadcast_to, chunk | elu, threshold, add.out |
| 是否有 Fallback | 是 | 是 | 是 |

---

## 11. 关键文件

| 文件 | 内容 |
|------|------|
| `native_functions.yaml` | 所有算子定义 |
| `BinaryOps.cpp` | Binary ops 的 Meta/Impl |
| `Activation.cpp` | Activation 类 ops（elu, threshold） |
| `TensorShape.cpp` | Shape 相关的 Composite ops |
| `ReduceOps.cpp` | Reduction ops 的 Composite 实现 |
| `FunctionalizeFallbackKernel.cpp` | Functionalization Fallback |
