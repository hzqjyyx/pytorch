这个文件是 PyTorch 自定义算子(custom operators)的编写指南，主要解决"如何为 PyTorch 添加新算子或新内核"的问题。

## 核心概念

**两种算子定义方式：**
1. `native_functions.yaml` - 用于 PyTorch 公共 API 的算子
2. Custom Operator API - 用于以下场景：
   - 非公共 API 的算子
   - 不想修改核心代码库（如开发独立的共享库）
   - C++ 扩展或内联 C++
   - 后端库（如 XLA/MAIA）为现有算子添加新内核

**使用 Custom Operator API 的代价：**
- 不会生成 C++ API（没有 `Tensor::your_op()` 或 `at::your_op()`）
- Python 调用方式不同：必须通过 `torch.ops.your_op()` 而非 `torch._C`
- 自动求导需要手动实现（使用 `torch::autograd::Function`）

## 编写方式

**1. 函数形式：**
```cpp
namespace { Tensor my_kernel_cpu(const Tensor& a, const Tensor& b) {...} }

static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op", torch::RegisterOperators::options()
       .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(CPU()));
```

**2. Lambda 形式：**
```cpp
static auto registry = torch::RegisterOperators()
    .op("my_namespace::my_op", torch::RegisterOperators::options()
        .kernel(CPU(), [] (const Tensor& a) -> Tensor{...}));
```
注意：lambda 必须无状态（无闭包）

**3. Catch-all 内核：**
```cpp
.op("my_namespace::my_op", torch::RegisterOperators::options()
    .catchAllKernel<decltype(my_kernel_fallback), &my_kernel_fallback>());
```
禁用分发机制，对所有后端都调用同一内核

## 高级特性

**多后端支持：**
```cpp
static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op", options().kernel<...>(CPU()))
   .op("my_namespace::my_op", options().kernel<...>(CUDA()));
```

**显式 Schema 定义：**
```cpp
.op("my_namespace::my_op(Tensor a, Tensor b) -> Tensor", ...)
.op("my_namespace::my_op(Tensor(a) x, int y = 3, int? z = None) -> Tensor(a|b)", ...)
```
支持注解、默认值等

**多返回值：**
```cpp
std::tuple<Tensor, int64_t, Tensor> my_kernel_cpu(...) {...}
```

**重载：**
```cpp
.op("my_namespace::my_op.overload1(Tensor a) -> Tensor", ...)
.op("my_namespace::my_op.overload2(Tensor a, Tensor b) -> Tensor", ...)
```

## 支持的类型

**输入/输出类型：**
- `at::Tensor`
- `double`（不支持 `float`）
- `int64_t`（不支持其他整数类型）
- `bool`
- `std::string_view`
- `at::Scalar`
- `std::optional<T>`
- `torch::List<T>`
- `torch::Dict<Key, Value>`

推荐按值传递所有参数（包括 Tensor），会被移动传递无性能损失。

## 调用方式

**从 PyTorch/JIT：**
```python
torch.ops.my_namespace.my_op(a, b)
```

**从 Caffe2：**
```cpp
C10_EXPORT_C10_OP_TO_CAFFE2_CPU(MyCaffe2OperatorName, "my_namespace::my_op")
C10_EXPORT_C10_OP_TO_CAFFE2_CUDA(MyCaffe2OperatorName, "my_namespace::my_op")
```

限制：
- 必须有 `Tensor` 或 `Tensor[]` 参数（不能同时有）
- 只支持 `int`、`double`、`bool` 基础类型
- 不支持列表或可选的基础类型
- 返回值必须是单个或多个 `Tensor`

---

**ROCm/Backward 相关（简要）：**
- 支持 ROCm 后端的内核注册（类似 CUDA）
- Backward 算子通过 `torch::autograd::Function` 手动实现
- 与分发机制尚未完全集成，需手动处理多后端情况
