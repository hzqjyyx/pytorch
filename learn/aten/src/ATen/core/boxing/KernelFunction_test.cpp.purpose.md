这是 PyTorch 的 `KernelFunction` 类的单元测试文件，主要测试内核函数的"装箱"（boxed）和"拆箱"（unboxed）调用机制。

## 核心概念

**装箱调用（Boxed Calling）**：
- 参数通过 `Stack`（`vector<IValue>`）传递
- 运行时类型检查和转换
- 统一的调用接口
- 函数签名：`void(const OperatorHandle&, Stack*)`

**拆箱调用（Unboxed Calling）**：
- 直接使用 C++ 原生类型调用
- 编译时类型安全
- 性能更优
- 函数签名：如 `int64_t(int64_t, int64_t)` 或 `void(int64_t, int64_t)`

## 测试结构

### 1. 测试用的内核实现（kernels 命名空间）

**装箱函数**：
- `boxed_func_with_return`：接收两个 int，返回常量 5
- `boxed_func_without_return`：接收两个 int，无返回值
- `boxed_func_with_multi_return`：返回两个值（a+b 和 a*b）

**拆箱函数**：
- `unboxed_functor_with_return`/`unboxed_functor_without_return`：函数对象
- `unboxed_function_with_return`/`unboxed_function_without_return`：普通函数
- `unboxed_lambda_with_return`/`unboxed_lambda_without_return`：lambda 函数

**In-place/Out-of-place 操作**：
- `boxed_func_for_inplace_op`：`(Tensor&, Scalar) -> Tensor&`，模拟 `tensor.add_(scalar)`
- `boxed_func_for_outofplace_op`：`(Scalar, Tensor&) -> Tensor&`，输出参数在后
- `boxed_func_for_outofplace_multi_op`：多个输出参数

### 2. 验证函数

**装箱调用验证**：
- `expectBoxedCallingWithReturnWorks`：验证通过 `callBoxed` 调用，检查栈上返回值
- `expectBoxedCallingWithoutReturnWorks`：验证空栈返回
- `expectBoxedCallingWithMultiReturnWorks`：验证栈上多个返回值

**拆箱调用验证**：
- `expectUnboxedCallingWithReturnWorks`：通过 `call<int64_t, int64_t, int64_t>` 调用
- `expectUnboxedCallingWithoutReturnWorks`：通过 `call<void, int64_t, int64_t>` 调用
- `expectUnboxedCallingWithMultiReturnWorks`：返回 `std::tuple<int64_t, int64_t>`

**In-place/Out-of-place 验证**：
- `expectInPlaceBoxedCallingWorks`：验证返回的 Tensor 与输入是同一个对象
- `expectOutOfPlaceBoxedCallingWorks`：验证输出参数被正确修改和返回
- `expectOutOfPlaceMultiUnboxedCallingWorks`：验证多个输出参数的结构化绑定

### 3. 测试用例组织

测试矩阵：**内核类型 × 调用方式 × 返回值类型**

**内核创建方式**：
- `makeFromBoxedFunction`：从装箱函数创建
- `makeFromUnboxedFunctor`：从函数对象创建
- `makeFromUnboxedFunction`：从编译时函数（`TORCH_FN` 宏）创建
- `makeFromUnboxedRuntimeFunction`：从运行时函数指针创建
- `makeFromUnboxedLambda`：从 lambda 创建

**测试覆盖**：
- 装箱函数 → 装箱调用 ✓
- 装箱函数 → 拆箱调用 ✓（自动转换）
- 拆箱函数/仿函数/lambda → 装箱调用 ✓（自动装箱）
- 拆箱函数/仿函数/lambda → 拆箱调用 ✓

## 关键机制

1. **双向兼容性**：装箱内核可以被拆箱调用，拆箱内核可以被装箱调用，`KernelFunction` 自动处理转换

2. **DispatchKeySet**：所有调用都需要传递 `CPU_TEST_SET`，这是调度系统的要求（见 Note [Plumbing Keys Through The Dispatcher]）

3. **类型安全**：
   - 装箱调用：运行时通过 `EXPECT_TRUE(stack->at(0).isInt())` 检查类型
   - 拆箱调用：编译时模板参数 `call<ReturnType, ArgTypes...>` 保证类型

4. **引用语义**：In-place/Out-of-place 操作测试验证了返回的 Tensor 引用与输入参数的同一性（`is_same` 检查）

---

**次要内容（简要）：**
- 无 ROCm 相关内容
- 无反向传播（Backward）相关内容
- `makeDummyOperatorHandle()`：创建虚拟算子句柄用于测试，注册为 `"my::dummy() -> ()"`
- `called_with_args`：全局变量，用于验证内核是否被正确调用并接收了预期参数
