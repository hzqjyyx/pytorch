## LegacyTypeDispatch.h 文件分析

这个文件定义了 ATen 中的**遗留类型分发机制**，主要包含四个 RAII 守卫类，用于控制操作符的分发行为：

### 核心概念
- ATen 的旧分发机制基于 Type 对象（一个虚函数分发表）
- 已被 ATenDispatch 和 c10 dispatcher 替代，但仍保留以支持特定场景

### 四个守卫类

**1. AutoDispatchBelowAutograd**
- 禁用所有自动求导相关的分发键
- 将 Variable 类型操作重定向到非 Variable 实现
- 内核实现专用，用户代码应改用 `c10::InferenceMode`

**2. AutoNonVariableTypeMode**
- 已弃用（将在 1.10 版本移除）
- 功能同 AutoDispatchBelowAutograd
- 触发警告提示开发者迁移

**3. AutoDispatchSkipFunctionalize**
- 跳过 Functionalize 分发键
- 绕过函数化逻辑

**4. AutoDispatchBelowADInplaceOrView**
- 禁用 Autograd 和 ADInplaceOrView 分发键
- 用于原地操作和视图操作的内核实现
- 确保不会递归回到相同分发键的内核

### 关键设计原理
- Variable/Tensor 合并后，不再有解包操作
- 需要这些守卫防止无限递归分发
- 所有守卫基于 `ExcludeDispatchKeyGuard` 机制

---

**主要用途总结：**
- 控制操作符分发的递归行为
- 在内核实现中绕过特定的分发层
- 防止自动求导和函数化逻辑的意外触发
