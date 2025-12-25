# ATen Operator Registration 核心功能

这两个文件实现了 PyTorch 的**算子注册系统**，允许开发者将算子实现（kernel）注册到 dispatcher 并建立 schema 与 kernel 的映射关系。

## 核心架构

### 1. RegisterOperators 类
- **RAII 管理器**：持有 `std::vector<RegistrationHandleRAII> registrars_`，析构时自动解注册
- **流式 API**：支持链式调用 `.op().op()...`
- **生命周期**：必须保持实例存活，否则注册失效

### 2. 注册流程

**入口**: `RegisterOperators::op(Options&& options)`
```
op() → checkSchemaAndRegisterOp_() → registerOp_()
```

**checkSchemaAndRegisterOp_() 的两条路径**:

```cpp
// 路径 1: 显式指定 schema
if (schemaOrName_ 是 FunctionSchema) {
    checkNoDuplicateKernels_();
    registerOp_();
}

// 路径 2: 仅指定算子名，需推断 schema
else {
    schema = inferSchemaFromKernels_();  // 从 kernel 函数签名推断
    检查不能使用 AliasAnalysisKind::FROM_SCHEMA;
    checkNoDuplicateKernels_();
    registerOp_();
}
```

**inferSchemaFromKernels_()**: 
- 遍历所有 kernel，找第一个带 `inferred_function_schema` 的
- 通过模板元编程从 C++ 函数签名推断参数/返回值类型

**checkNoDuplicateKernels_()**: 
- 防止同一 DispatchKey 注册多个 kernel
- 防止注册多个 catch-all kernel

**registerOp_()**: 
- 调用 `Dispatcher::singleton().registerDef()` 注册 schema
- 调用 `Dispatcher::singleton().registerImpl()` 注册每个 kernel
- 所有 handle 存入 `registrars_` 向量

## 注册方式

### Schema 指定
```cpp
.schema("my_op(Tensor a) -> Tensor")          // 完整 schema
.schema("my_op")                              // 仅名称，推断签名
.schema(FunctionSchema(...))                  // 内部 API
```

### Kernel 注册（按 DispatchKey）

**1. Functor (继承 OperatorKernel)**
```cpp
.kernel<MyKernelCPU>(DispatchKey::CPU, constructor_args...)
.catchAllKernel<MyKernelCPU>(constructor_args...)
```
- 要求: `std::is_base_of_v<OperatorKernel, KernelFunctor>`
- 包装为: `KernelFunction::makeFromUnboxedFunctor<false, KernelFunctor>`

**2. 函数指针（编译期）**
```cpp
.kernel<decltype(my_func), &my_func>(DispatchKey::CPU)
.catchAllKernel<decltype(my_func), &my_func>()
```
- 包装为: `KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernel_func))`

**3. 函数指针（运行期）**
```cpp
.kernel(DispatchKey::CPU, my_func_ptr)
.catchAllKernel(my_func_ptr)
```
- 包装为: `KernelFunction::makeFromUnboxedRuntimeFunction(kernel_func)`

**4. 无状态 Lambda**
```cpp
.kernel(DispatchKey::CPU, [](Tensor a) -> Tensor {...})
.catchAllKernel([](Tensor a) -> Tensor {...})
```
- 检查: `guts::is_stateless_lambda` (禁止捕获)
- 包装为: `KernelFunction::makeFromUnboxedLambda`

**5. Boxed Kernel (内部 API)**
```cpp
.kernel<&boxed_kernel_func>(DispatchKey::CPU)
```
- 用于 stack-based 调用

### 简化 API

```cpp
// 直接传函数指针/lambda，自动推断为 catch-all kernel
RegisterOperators()
  .op("my_op", &my_kernel_cpu)
  .op("my_op", [](Tensor a) {...});
```

## 关键设计

### 1. Schema 推断
- 通过 `detail::inferFunctionSchemaFromFunctor<KernelFunctor>()` 实现
- 移除 `DispatchKeySet` 参数（仅内部使用，对 JIT 不可见）
- 使用 `inferFunctionSchemaFlattenedReturns` 处理返回值

### 2. Dispatch Key 机制
- **有 DispatchKey**: 仅匹配该 backend 的输入调用此 kernel
- **无 DispatchKey (catch-all)**: 禁用 dispatch，所有输入调用此 kernel

### 3. Alias Analysis
```cpp
.aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
```
- 不能与推断 schema 同时使用（推断无 aliasing 信息）
- 直接写入 `FunctionSchema.setAliasAnalysis()`

### 4. 类型安全
- 编译期检查:
  - Functor 必须继承 `OperatorKernel`
  - Lambda 必须无捕获
  - 函数不能是 `BoxedKernelFunction`（除非内部 API）
- `static_assert` 提供清晰错误信息

### 5. Move 语义
- `Options` 对象禁止拷贝/移动（`= delete`）
- 只能通过 `std::move(*this)` 转移所有权
- 强制单次使用，防止重复注册

## 数据结构

### KernelRegistrationConfig
```cpp
struct KernelRegistrationConfig {
    std::optional<DispatchKey> dispatch_key;
    KernelFunction func;                          // 包装后的可调用对象
    std::optional<impl::CppSignature> cpp_signature;
    std::unique_ptr<FunctionSchema> inferred_function_schema;
};
```

### Options 成员
```cpp
std::optional<std::variant<OperatorName, FunctionSchema>> schemaOrName_;
std::vector<KernelRegistrationConfig> kernels;
std::optional<AliasAnalysisKind> aliasAnalysisKind_;
```

## 典型使用示例

```cpp
// 示例 1: 显式 schema + 多 backend kernel
static auto registry = c10::RegisterOperators()
  .op(c10::RegisterOperators::options()
      .schema("my_op(Tensor a, Tensor b) -> Tensor")
      .kernel<MyKernelCPU>(DispatchKey::CPU)
      .kernel<MyKernelCUDA>(DispatchKey::CUDA));

// 示例 2: 推断 schema + lambda
static auto registry = c10::RegisterOperators()
  .op("my_op", [](Tensor a) -> Tensor { return a + 1; });

// 示例 3: 构造函数注册
static auto registry = c10::RegisterOperators(
  "my_op", &my_kernel_cpu);
```

## 错误检查

1. **Schema 重复指定**: 多次调用 `.schema()` 抛异常
2. **Alias Analysis 重复**: 多次调用 `.aliasAnalysis()` 抛异常
3. **重复 DispatchKey**: `checkNoDuplicateKernels_()` 检测
4. **多个 catch-all**: 只允许一个无 DispatchKey 的 kernel
5. **推断失败**: 所有 kernel 都无 `inferred_function_schema` 时报错
6. **空 kernel 列表**: 推断时要求至少一个 kernel

---

### ROCm/Backward 相关
- ROCm: 支持 `DispatchKey::HIP` 注册 AMD GPU kernels
- Backward: 可注册 `DispatchKey::Autograd*` 相关的反向传播 kernels
