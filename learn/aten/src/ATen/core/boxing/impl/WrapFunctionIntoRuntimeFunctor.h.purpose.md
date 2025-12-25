这个文件定义了一个模板类 `WrapFunctionIntoRuntimeFunctor`，用于将任意运行时函数对象包装成继承自 `c10::OperatorKernel` 的内核对象。

**主要结构：**

- **`WrapFunctionIntoRuntimeFunctor_` 模板类**：核心实现类，通过模板特化处理函数类型和参数列表
  - 存储函数对象（`kernel_func_`）
  - 实现 `operator()` 转发调用到被包装的函数

- **`WrapFunctionIntoRuntimeFunctor` 别名**：对外接口，自动推导函数特征
  - 使用 `guts::infer_function_traits_t` 提取返回类型和参数类型
  - 简化用户使用，无需手动指定模板参数

**支持的函数类型：**

- Lambda 函数
- 函数对象（Functors）
- 函数指针

**关键特点：**

- 继承 `c10::OperatorKernel` 使其能作为 c10 框架内核使用
- 完美转发参数（`std::forward`）保持参数的值类别
- 函数指针调用有额外的运行时开销
