## TorchDispatchUtils 功能分析

**文件用途：**

这个模块提供了用于检查张量是否启用了 TorchDispatch 机制的工具函数。TorchDispatch 是 PyTorch 的动态分派系统，允许自定义张量子类和操作重载。

**核心实现：**

`tensor_has_dispatch()` 函数检查单个张量的 DispatchKeySet 是否包含 `Python` 或 `PythonTLSSnapshot` 这两个分派键。如果张量有这些键，说明它已被注册为需要通过 Python 层处理的自定义张量。

`tensorlist_has_dispatch()` 有两个重载版本：
- 第一个版本接受 `ITensorListRef`，遍历列表中的每个张量
- 第二个版本接受 `c10::List<std::optional<at::Tensor>>`（包含可选张量的列表），需要检查空值

两个列表版本都是短路求值：只要找到一个张量启用了 dispatch，就立即返回 true。

**关键设计：**

- 头文件声明三个公共 API，使用 `TORCH_API` 宏导出
- 实现文件在 `at::impl` 命名空间中定义具体逻辑
- 利用 DispatchKeySet 的 `has_any()` 方法进行高效的位级别检查

**主要功能：**

- 张量级别检查：判断单个张量是否需要 TorchDispatch 处理
- 列表级别检查：快速判断张量列表中是否存在需要分派的张量
- 支持可选张量：正确处理包含 null 值的张量列表
- 分派决策：为运行时选择执行路径提供依据（例如决定是否调用 Python 回调）
