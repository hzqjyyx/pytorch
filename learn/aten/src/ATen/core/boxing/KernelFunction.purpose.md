## KernelFunction 的主要功能

`KernelFunction` 是 PyTorch 中的核心调度机制，用于统一管理两种不同的内核调用方式：**boxed** 和 **unboxed**。

### 核心设计

- **Boxed 形式**：所有参数打包成一个栈（Stack），函数签名统一为 `void(OperatorKernel*, DispatchKeySet, Stack*)`
- **Unboxed 形式**：直接传递类型化的参数，如 `Tensor unboxed_func(Tensor a, bool b)`
- **自动转换**：如果以一种方式创建，可以以另一种方式调用，KernelFunction 会自动处理 boxing/unboxing 开销

### 主要功能模块

**1. 内核创建工厂方法**
- `makeFromBoxedFunction()` - 从 boxed 函数创建
- `makeFromUnboxedFunction()` - 从 unboxed 函数创建（编译时已知）
- `makeFromUnboxedRuntimeFunction()` - 从 unboxed 函数创建（运行时）
- `makeFromUnboxedLambda()` - 从 unboxed lambda 创建
- `makeFromUnboxedFunctor()` / `makeFromBoxedFunctor()` - 从 functor 创建
- `makeFromBoxedKernel()` - 从 BoxedKernel 创建

**2. 调用方法**
- `callBoxed()` - 以 boxed 方式调用
- `call<Return, Args...>()` - 以 unboxed 方式调用

**3. 验证和状态方法**
- `isValidUnboxed()` - 检查是否有有效的 unboxed 形式
- `isValidSymUnboxed()` - 检查是否支持符号整数（SymInt）
- `isValid()` - 检查总体有效性
- `isFallthrough()` - 检查是否为 fallthrough 类型
- `dumpState()` - 打印当前状态

**4. 特殊内核**
- `fallthrough_kernel` - 重新分发到下一个 dispatch key（用于跳过当前处理）
- `ambiguous_autogradother_kernel` - 处理冲突：CompositeImplicitAutograd vs AutogradOther
- `named_not_supported_kernel` - 拒绝对 named tensors 的支持

### SymInt 支持

提供了模板元编程机制（`has_symint`, `remove_symint`, `fn_remove_symint`）来处理符号整数的自动转换，支持在有/无 SymInt 的两种版本间自动转换。

### 内部存储

```cpp
BoxedKernel boxed_kernel_func_;           // boxed 形式
void* unboxed_kernel_func_;               // unboxed 形式
void* sym_unboxed_kernel_func_;           // 支持 SymInt 的 unboxed 形式
```

### 关键特性总结

- **透明转换**：自动在 boxed/unboxed 之间转换
- **性能优化**：支持编译时已知函数指针以启用内联
- **快速路径**：优先使用 unboxed 形式避免装箱开销
- **灵活创建**：支持函数指针、functor、lambda 等多种内核类型
- **调度集成**：与 PyTorch 的 dispatcher 深度集成，支持 dispatch key 管理
