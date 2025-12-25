## 文件功能分析

这个文件实现了 PyTorch 的**选择性构建(Selective Build)**机制，用于在编译时只包含指定的操作符，以减小二进制文件大小。

### 核心机制

1. **构建时操作符过滤**
   - 通过 `-DTORCH_OPERATOR_WHITELIST="aten::add;aten::sub"` 编译标志指定允许列表
   - 在编译阶段移除未列出的操作符注册调用
   - 链接器随后删除所有未被注册的操作符函数

2. **主要检查函数**

   - `allowlist_contains()`: 在分号分隔的允许列表中查找特定项
   - `op_allowlist_check()`: 验证操作符是否应被注册
   - `schema_allowlist_check()`: 验证操作符模式(schema)是否应被注册
   - `custom_class_allowlist_check()`: 验证自定义类是否应被注册

3. **构建功能检查**
   - `is_build_feature_available()`: 检查构建功能是否可用
   - `BUILD_FEATURE_REQUIRED()` 宏: 在运行时断言所需功能可用，否则抛出异常
   - `BUILD_FEATURE_AVAILABLE()` 宏: 查询功能可用性

4. **两种工作模式**

   - **选择性构建模式**: 根据允许列表决定功能可用性
   - **检测模式** (ENABLE_RECORD_KERNEL_FUNCTION_DTYPE): 始终返回 true，用于追踪和分析

### 关键特性

- **编译时计算**: 所有检查函数使用 `constexpr`，编译器可执行死代码消除
- **限制**: 仅当调度键/操作符名称在编译时明显可见时才有效
- **支持过载**: 允许列表记录操作符而非过载，包含某操作符时其所有过载都被包含

---

### 快速总结

- **目的**: 精简化 PyTorch 二进制，移除未使用的操作符
- **方法**: 编译时过滤 + 链接器修剪
- **API**: 提供操作符/自定义类/构建功能的允许列表检查
- **模式**: 支持选择性构建和检测两种模式
