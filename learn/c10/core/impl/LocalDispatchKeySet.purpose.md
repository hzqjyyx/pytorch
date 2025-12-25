## LocalDispatchKeySet 核心功能

这两个文件实现了PyTorch的**线程本地分发键集管理系统**，用于控制操作在运行时的分发路径。

### 核心概念

**DispatchKeySet** 是一个集合，包含两个互相独立的部分：

1. **included_ (包含集)** - 添加额外的分发键供考虑
   - 例如：添加 Profiling 键来启用所有张量操作的性能分析
   
2. **excluded_ (排除集)** - 禁止特定分发键参与分发
   - 例如：在处理完 Variable 后排除 Autograd，防止重复处理
   - **排除优先级更高** - 如果键同时在两个集中，排除生效

### 实现细节

**PODLocalDispatchKeySet** (Plain Old Data) 是TLS存储的实际类型：
- 只包含两个 `uint64_t` 字段，满足零初始化要求
- 通过与 `c10::default_included_set` 和 `c10::default_excluded_set` 进行XOR操作，实现非零状态的有效编码
- 注释说明：某些Windows编译器要求TLS必须零初始化

**LocalDispatchKeySet** 是用户facing的包装类：
- 持有两个 DispatchKeySet 对象
- 可从 PODLocalDispatchKeySet 隐式转换

### API 分类

**RAII 风格（推荐）** c10/core/impl/LocalDispatchKeySet.cpp:48-72：
- `IncludeDispatchKeyGuard` - 作用域内临时包含某键
- `ExcludeDispatchKeyGuard` - 作用域内临时排除某键
- `ForceDispatchKeyGuard` - 完全保存/恢复整个分发状态
- 优点：自动清理，防止遗漏

**非RAII风格** c10/core/impl/LocalDispatchKeySet.cpp:78-116：
- `tls_is_dispatch_key_excluded/included()` - 查询单个键状态
- `tls_set_dispatch_key_excluded/included()` - 设置单个键状态
- `tls_is_dispatch_keyset_excluded/included()` - 查询键集状态
- 用途：跨越多个Python→C++调用时保持状态（如Python上下文管理器）
- 缺点：效率略低（每次调用都需TLS查询）

### 关键实现细节

- **线程局部存储**：`thread_local PODLocalDispatchKeySet raw_local_dispatch_key_set` c10/core/impl/LocalDispatchKeySet.cpp:14
- **平台差异处理**：Windows/Android/iPhone 使用函数包装器而非直接访问
- **状态转换优化**：Guard 对象缓存指针避免重复TLS查询

---

• 管理线程本地的分发键集（included 和 excluded）
• 通过XOR编码实现TLS零初始化约束下的非零默认状态
• 提供RAII风格的Guard类进行作用域绑定的状态修改
• 提供非RAII API用于跨调用边界保持状态
• 支持查询单个键或整个键集的包含/排除状态
