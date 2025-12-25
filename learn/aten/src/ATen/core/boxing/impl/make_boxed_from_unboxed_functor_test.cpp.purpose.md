这是一个针对 PyTorch 算子注册系统的单元测试文件，专门测试基于 Functor（函数对象）的 kernel 注册机制。

## 核心测试内容

### 1. 基本 Functor Kernel 注册与调用
- 测试将 `OperatorKernel` 子类（如 `IncrementKernel`、`DecrementKernel`）注册为算子实现
- 验证注册后能正确通过 Dispatcher 查找 schema 并调用
- 支持为不同 DispatchKey（CPU/CUDA）注册不同的 kernel 实现

### 2. 多种输入输出类型支持
**输出类型测试：**
- 无返回值 (`void`)
- 零返回值 (`std::tuple<>`)
- 单一返回值（int、Tensor）
- 列表返回值（`c10::List<Tensor>`、`c10::List<int64_t>`）
- 多返回值（`std::tuple` 包含 Tensor、int、List、Optional、Dict）
- Dict 返回值

**输入类型测试：**
- Tensor 输入（按引用/按值）
- 基础类型输入（int64_t）
- 列表输入（`c10::List<int64_t>`、`c10::List<Tensor>`）
- Dict 输入（`Dict<string, Tensor>`、`Dict<string, string>`）
- Tuple 输入
- Optional 输入（`std::optional<Tensor>`、`std::optional<int64_t>`、`std::optional<string>`）

### 3. 高级特性测试
- **有状态 Kernel**：`KernelWithCache` 测试 functor 内部状态在多次调用间保持
- **构造函数参数**：测试 kernel 在注册时传入构造参数（单个/多个参数），不同 DispatchKey 使用不同配置
- **Fallback Kernel**：使用 `catchAllKernel()` 注册，支持没有 Tensor 参数的算子
- **Schema 推断**：不显式指定 schema 时，从 functor 的 `operator()` 签名自动推断

### 4. 错误检测测试
验证注册时的类型检查机制，当 schema 与 functor 签名不匹配时抛出异常：
- 参数数量不匹配
- 参数类型不匹配
- 返回值数量不匹配
- 返回值类型不匹配

### 5. Boxed 与 Unboxed 调用
- 大部分测试使用 boxed 调用（通过 `Stack` 传递参数）
- `expectCallsConcatUnboxed` 测试直接的类型化调用（`callOpUnboxed<ReturnType, Args...>`）

## 测试模式
- 使用 Google Test 框架（`TEST` 宏）
- 通过 `RegisterOperators` API 注册临时算子（测试结束后自动清理）
- 使用 `_test::` 命名空间避免与真实算子冲突
- 使用 `dummyTensor(DispatchKey)` 创建测试用 Tensor
- 通过 `c10::Dispatcher::singleton().findSchema()` 查找并调用算子

## 其他测试点
- 多个算子在同一/不同 registrar 中注册
- Tensor 输入按引用/按值传递的不同方式
- Optional 参数的存在/缺失情况处理
