## GradMode 文件功能分析

这两个文件构成了 PyTorch 的梯度模式管理系统，用于控制自动求导（autograd）的行为。

**GradMode.h** 定义了四个主要的 RAII 结构体：

1. **GradMode** - 静态接口，用于查询和设置全局梯度模式状态。通过 `AutogradState::get_tls_state()` 访问线程局部存储（TLS）中的梯度模式标志。

2. **AutoGradMode** - RAII 守卫类，构造时保存当前梯度模式并设置新值，析构时恢复原值。禁用了拷贝和移动语义，确保栈上的 RAII 语义。

3. **NoGradGuard** - `AutoGradMode` 的子类，直接将梯度模式禁用（相当于 `AutoGradMode(false)`），用于阻止操作链中的梯度计算。

4. **AutoFwGradMode** - 类似的 RAII 守卫，但操作的是前向梯度模式（forward mode automatic differentiation），而非反向梯度模式。

**GradMode.cpp** 是实现文件，包含两个简单的函数：
- `is_enabled()` - 查询当前梯度模式状态
- `set_enabled(bool)` - 设置梯度模式状态

所有状态管理都委托给 `AutogradState` 的线程局部存储。

---

**核心功能总结：**

- 提供线程安全的梯度模式全局开关
- RAII 模式确保嵌套作用域内的梯度状态自动恢复
- 支持反向梯度模式和前向梯度模式的独立控制
- `NoGradGuard` 提供便捷接口禁用梯度计算（用于推理、评估等场景）
