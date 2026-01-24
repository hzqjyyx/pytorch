# Structured Kernel 机制详解

## 1. 核心概念

### 1.1 什么是 Structured Kernel

Structured kernel 是 PyTorch 的一种算子实现模式，将算子分为两个独立的部分：
- **Meta 函数**：计算输出张量的 shape、dtype、strides 等元信息
- **Impl 函数**：执行实际的计算逻辑

### 1.2 两个关键字段

| 字段 | 用于 | 含义 |
|------|------|------|
| `structured: True` | `.out` 变体 | 标记该函数为 structured kernel 的定义者 |
| `structured_delegate: xxx.out` | functional/inplace 变体 | 委托给指定的 structured kernel |

---

## 2. YAML 定义示例

### 2.1 sgn 算子（一元操作）

**位置**: `aten/src/ATen/native/native_functions.yaml:428-454`

```yaml
# Functional 变体 - 委托给 sgn.out
- func: sgn(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: sgn.out
  dispatch:
    SparseCPU, SparseCUDA: sgn_sparse
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_sgn
  tags: pointwise

# Inplace 变体 - 也委托给 sgn.out
- func: sgn_(Tensor(a!) self) -> Tensor(a!)
  variants: method
  structured_delegate: sgn.out
  dispatch:
    SparseCPU, SparseCUDA: sgn_sparse_
  tags: pointwise

# Out 变体 - 核心定义，标记为 structured
- func: sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: sgn_out
    MPS: sgn_out_mps
  tags: pointwise
```

### 2.2 add 算子（二元操作）

**位置**: `aten/src/ATen/native/native_functions.yaml:554-591`

```yaml
# Tensor overload - 委托
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck
  structured_delegate: add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMeta: add_sparse
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: [core, pointwise]

# Inplace 变体
- func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
  device_check: NoCheck
  variants: method
  structured_delegate: add.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMeta: add_sparse_
    MkldnnCPU: mkldnn_add_
  tags: pointwise

# Out 变体 - 核心定义
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck
  structured: True
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

---

## 3. 代码生成中的处理

### 3.1 字段定义

**位置**: `torchgen/model.py:544-561`

```python
# NativeFunction 类中的字段定义

# Whether or not this out functions is a "structured kernel". Structured
# kernels are defined a little differently from normal kernels; in
# particular, their shape checking logic is defined separately from
# the kernel. Only out functions can be structured; other functions
# delegate to the out function using the structured_delegate keyword.
structured: bool

# Whether or not this non-out function is a structured kernel, defined
# in terms of the out kernel referenced by the string here.
structured_delegate: OperatorName | None

# Only valid for structured kernels. Specifies alternative of what
# to inherit from when defining the meta class for the structured
# operator. This will usually be TensorIteratorBase.
structured_inherits: str | None
```

### 3.2 验证规则

**位置**: `torchgen/model.py:997-1026`

```python
# Rule 1: structured 只能在 out 变体上
if self.structured:
    assert self.func.kind() == SchemaKind.out, (
        "Put structured field on the out= variant of a function; "
        "did you mean structured_delegate?"
    )

# Rule 2: structured_delegate 不能在 out 变体上
if self.structured_delegate:
    assert self.func.kind() != SchemaKind.out, (
        "structured_delegate field not allowed on out= functions; "
        "did you mean structured?"
    )

# Rule 3: 两个字段互斥
assert not (self.structured and self.structured_delegate), (
    "Cannot have both structured and structured_delegate on function"
)

# Rule 4: structured_inherits 需要 structured: True
if self.structured_inherits is not None:
    assert self.structured, (
        "structured_inherits must also imply structured: True"
    )
```

### 3.3 NativeFunctionsGroup 验证

**位置**: `torchgen/model.py:1153-1166`

```python
if self.structured:
    # 验证 functional 变体的 structured_delegate 指向 out 变体
    assert self.functional.structured_delegate == self.out.func.name, (
        f"{self.functional.func.name} delegates to "
        f"{self.functional.structured_delegate} "
        f"but its actual delegate is {self.out.func.name}"
    )
    # inplace 变体也必须 delegate 到同一个 out 变体
    if self.inplace is not None:
        assert self.inplace.structured_delegate == self.out.func.name
```

---

## 4. 生成的代码结构

### 4.1 Structured Kernel 类

**位置**: `torchgen/dest/register_dispatch_key.py:820-858`

生成的代码模式：

```cpp
// 对于 functional 变体
{class_name} op;

// 对于 inplace 变体
{class_name} op(self);

// 对于 out 变体
{class_name} op({out_args});

// Meta 函数调用
auto precompute = op.meta({meta_args});

// Impl 函数调用
op.impl({impl_args});
```

### 4.2 set_output 方法

**位置**: `torchgen/dest/register_dispatch_key.py:610-660`

生成三种 set_output 变体：

```cpp
// 严格步幅匹配
void set_output_strided(
    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
    TensorOptions options, DimnameList names
) override {
    // 对于 functional: 创建新输出
    // 对于 inplace: 检查原地操作合法性
    // 对于 out: 调整输出张量大小/步幅
}

// 允许任意步幅
void set_output_raw_strided(
    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides_hint,
    TensorOptions options, DimnameList names
) override { ... }

// 强制连续步幅
void set_output_contiguous(
    int64_t output_idx, IntArrayRef sizes,
    TensorOptions options, DimnameList names
) { ... }
```

---

## 5. Meta 和 Impl 宏

### 5.1 宏定义

**位置**: `aten/src/ATen/TensorMeta.h:14-58`

```cpp
// Meta 函数定义宏
#define TORCH_META_FUNC(name) \
  void structured_##name::meta

#define TORCH_META_FUNC2(name, overload) \
  void structured_##name##_##overload::meta

// Impl 函数定义宏
#define TORCH_IMPL_FUNC(name) \
  void structured_##name::impl
```

### 5.2 使用示例

```cpp
// Meta 函数
TORCH_META_FUNC(sgn)(const Tensor& self) {
    // 计算输出形状、dtype 等
    set_output(self.sizes(), self.options());
}

// Impl 函数
TORCH_IMPL_FUNC(sgn_out)(const Tensor& self, const Tensor& result) {
    // 实际的计算逻辑
    sgn_stub(device_type(), *this);
}
```

---

## 6. MetaBase 接口

**位置**: `aten/src/ATen/TensorMeta.h:60-133`

```cpp
struct TORCH_API MetaBase {
    // 输出设置方法选择决策树：
    //
    // 内核支持任意步幅？
    // ├─ YES → set_output_raw_strided()
    // └─ NO → 步幅需要连续？
    //     ├─ YES → set_output_contiguous()
    //     └─ NO → set_output_strided()

    virtual void set_output_strided(...) { ... }
    virtual void set_output_raw_strided(...) { ... }
    void set_output_contiguous(...) { ... }

    // 获取设备类型
    c10::DeviceType device_type() const;

    // 获取可能的输出
    const Tensor& maybe_get_output(int64_t output_idx = 0);
};
```

---

## 7. 完整调用流程

以 `add.Tensor` 调用为例：

```
用户代码:
t1.add(t2, alpha=2)
    ↓
torch.ops.aten.add.Tensor(t1, t2, alpha=2)
    ↓
Dispatcher 查询
    ↓
发现 structured_delegate: add.out
    ↓
查询 add.out 的实现
    ↓
// 生成的代码:
at::meta::structured_add op;           // 创建 structured kernel 对象
op.meta(t1, t2, alpha);                // 调用 meta，计算 shape，设置输出
op.impl(result, t1, t2, alpha);        // 调用 impl，执行实际计算
return result;
```

---

## 8. 语义总结

### `structured: True`

- **应用于**：Out 变体（`.out`）
- **含义**：该函数是 structured kernel 的核心定义
- **效果**：
  - 实现分为 meta 和 impl 两部分
  - 自动生成 `at::meta::structured_<op_name>` 类
  - 可以指定 `structured_inherits` 继承基类

### `structured_delegate: xxx.out`

- **应用于**：Functional/Inplace 变体
- **含义**：将实现委托给指定的 structured out 变体
- **效果**：
  - 不需要单独实现，复用 out 变体的 meta/impl
  - 自动生成包装代码
  - Dispatch 表查询时会转到 out 变体

---

## 9. 优势

1. **代码复用**：三个变体共享同一套 meta/impl
2. **一致性**：Shape 检查逻辑集中在一处
3. **可维护性**：修改一次即可影响所有变体
4. **符号执行**：Meta 函数可独立运行，用于 shape 推导
5. **多后端支持**：Meta 共享，Impl 可以有多个特化版本

---

## 10. 关键文件

| 文件 | 内容 |
|------|------|
| `torchgen/model.py:544-561` | 字段定义 |
| `torchgen/model.py:670-684` | YAML 解析 |
| `torchgen/model.py:997-1026` | 验证规则 |
| `torchgen/gen.py:286-296` | 代码生成检查 |
| `torchgen/dest/register_dispatch_key.py:363-390` | Dispatch 生成 |
| `aten/src/ATen/TensorMeta.h` | Meta 宏定义 |
| `aten/src/ATen/native/native_functions.yaml` | YAML 定义 |
