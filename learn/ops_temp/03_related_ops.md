# 相关算子之间的联系

## 1. 概述

PyTorch 中相关的算子（如 `add`/`add_`/`add.out`）通过 `structured_delegate` 机制建立联系，共享核心实现代码。

---

## 2. add 算子的完整定义

**位置**: `aten/src/ATen/native/native_functions.yaml:554-632`

### 2.1 Functional 变体 (add.Tensor)

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck
  structured_delegate: add.out          # 关键：委托给 add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMeta: add_sparse
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: [core, pointwise]
```

### 2.2 Inplace 变体 (add_.Tensor)

```yaml
- func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
  device_check: NoCheck
  variants: method
  structured_delegate: add.out          # 关键：也委托给 add.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMeta: add_sparse_
    MkldnnCPU: mkldnn_add_
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_add__Tensor
  tags: pointwise
```

### 2.3 Out 变体 (add.out) - 核心实现

```yaml
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck
  structured: True                      # 关键：标记为 structured
  structured_inherits: TensorIteratorBase
  ufunc_inner_loop:
    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)
    ScalarOnly: add (Bool)
  dispatch:
    SparseCPU, SparseMeta: add_out_sparse_cpu
    SparseCUDA: add_out_sparse_cuda
    MkldnnCPU: mkldnn_add_out
    MPS: add_out_mps
  tags: pointwise
```

### 2.4 Scalar 变体

```yaml
# Scalar 版本使用 CompositeExplicitAutograd
- func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: add    # 组合实现

- func: add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
  variants: method
  dispatch:
    CompositeExplicitAutograd: add_
  autogen: add.Scalar_out             # 自动生成 out 变体
```

---

## 3. 三件套模式

### 3.1 变体说明

| 变体 | 签名特征 | 行为 | 内存 |
|------|---------|------|------|
| **Functional** | `add(Tensor, Tensor) -> Tensor` | 返回新张量 | 分配新内存 |
| **Inplace** | `add_(Tensor!, Tensor) -> Tensor!` | 修改第一个参数 | 复用输入内存 |
| **Out** | `add.out(..., Tensor! out)` | 写入预分配输出 | 使用提供的内存 |

### 3.2 委托关系图

```
add.Tensor ────┐
               ├───→ add.out (structured kernel)
add_.Tensor ──-┘          │
                          ├── meta(): 计算 shape、dtype
                          │
                          └── impl(): 调用 add_stub
                                      │
                              ┌───────┴───────┐
                              ↓               ↓
                            CPU             CUDA
                       add_kernel      add_kernel_cuda
```

---

## 4. 代码生成机制

### 4.1 数据模型

**位置**: `torchgen/model.py:545-555, 674-684`

```python
# NativeFunction 数据类中的关键字段
structured: bool                      # 该 out 函数是否为 structured kernel
structured_delegate: OperatorName | None  # 非 out 函数指向的 structured out 函数

# 解析 YAML 时的处理
structured_delegate_s = e.pop("structured_delegate", None)
if structured_delegate_s is not None:
    structured_delegate = OperatorName.parse(structured_delegate_s)
```

### 4.2 验证逻辑

**位置**: `torchgen/gen.py:286-296`

```python
for f in funcs:
    if f.structured_delegate is not None:
        delegate_func = func_map.get(f.structured_delegate)
        # 检查目标存在
        assert delegate_func is not None, (
            f"{f.func.name} is marked as a structured_delegate pointing to "
            f"{f.structured_delegate}, but {f.structured_delegate} is missing."
        )
        # 检查目标是 structured
        assert delegate_func.structured, (
            f"{f.func.name} is marked as a structured_delegate pointing to "
            f"{f.structured_delegate}, but {f.structured_delegate} is not "
            f"marked as structured."
        )
```

---

## 5. 生成的代码

### 5.1 Operators.h 中的结构体

**位置**: `torchgen/gen.py:607-694`

```cpp
struct add_Tensor {
  using schema = Tensor (const Tensor &, const Tensor &, const Scalar &);
  static constexpr const char* name = "aten::add";
  static constexpr const char* overload_name = "Tensor";

  // 调用分发器
  static Tensor call(const Tensor& self, const Tensor& other, const Scalar& alpha);

  // 重新分发（用于自定义 dispatch key）
  static Tensor redispatch(DispatchKeySet keySet, const Tensor& self,
                           const Tensor& other, const Scalar& alpha);
};
```

### 5.2 Functions.h 中的包装函数

**位置**: `torchgen/gen.py:720-725`

```cpp
// aten::add.Tensor
inline Tensor add(const Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
    return at::_ops::add_Tensor::call(self, other, alpha);
}
```

### 5.3 实际实现

**位置**: `aten/src/ATen/native/BinaryOps.cpp:151-156`

```cpp
// Meta 函数
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}
```

---

## 6. 运行时调用流程

### 6.1 用户调用 torch.add

```
1. torch.add(a, b)
   ↓
2. at::_ops::add_Tensor::call(a, b, 1)
   ↓
3. Dispatcher 查找 add.Tensor
   ↓
4. 发现 structured_delegate: add.out
   ↓
5. 创建 structured kernel 对象
   ↓
6. op.meta(a, b, 1)  // 计算输出 shape
   ↓
7. op.impl(result, a, b, 1)  // 执行计算
   ↓
8. 返回 result
```

### 6.2 ZeroTensor 的例子

**位置**: `BinaryOps.cpp:1050-1079`

展示了如何通过 redispatch 改变 dispatch key：

```cpp
static Tensor maybe_add_maybe_sub(const Tensor& self, const Tensor& other,
                                   const Scalar& alpha) {
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);

  // 显式调用 add_Tensor，但通过 Meta dispatch key
  auto meta_out = at::_ops::add_Tensor::redispatch(
      meta_dks, self.to(device_), other.to(device_), alpha);

  // ... 后续处理
}
```

---

## 7. 特殊分发的独立实现

某些后端不走 structured kernel 路径，有完全独立的实现：

```yaml
dispatch:
  SparseCPU, SparseCUDA: add_sparse     # 稀疏张量独立实现
  MkldnnCPU: mkldnn_add                  # MKL-DNN 独立实现
  NestedTensorCPU: NestedTensor_add     # 嵌套张量独立实现
```

这些函数直接在对应目录中实现，不使用 stub 机制。

---

## 8. 关系图总结

```
                    ┌─────────────────────────────────────────┐
                    │           native_functions.yaml         │
                    └─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
            add.Tensor            add_.Tensor           add.out
        (structured_delegate)  (structured_delegate)  (structured: True)
                    │                   │                   │
                    └───────────────────┴───────────────────┘
                                        │
                                        ↓
                              structured kernel 对象
                                        │
                         ┌──────────────┴──────────────┐
                         ↓                             ↓
                      meta()                        impl()
                   (计算 shape)                   (调用 stub)
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ↓                 ↓                 ↓
                                  CPU              CUDA               MPS
                             add_kernel      add_kernel_cuda     add_out_mps


                    ┌─────────────────────────────────────────┐
                    │              特殊后端分发                │
                    └─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
               add_sparse         mkldnn_add        NestedTensor_add
            (SparseCPU/CUDA)      (MkldnnCPU)     (NestedTensorCPU/CUDA)
                    │                   │                   │
                    └───────────────────┴───────────────────┘
                              (完全独立的实现)
```

---

## 9. 关键特性

| 特性 | 说明 |
|------|------|
| `structured: True` | 标记 out 变体为结构化内核 |
| `structured_delegate` | 使其他变体委托给 out 变体 |
| `ufunc_inner_loop` | 定义 CPU/GPU 上的具体计算循环 |
| `dispatch` 表 | 支持特殊后端的自定义实现 |
| `TensorIterator` | 处理广播、类型提升等通用逻辑 |

这个设计实现了：
- **代码复用**：三个变体共享核心实现
- **关注点分离**：meta 处理 shape，impl 处理计算
- **后端特化**：允许特殊后端有独立实现
