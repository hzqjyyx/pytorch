## VmapInterpreter 功能分析

这两个文件实现了 PyTorch functorch 中的 Vmap（向量化映射）解释器。

**VmapInterpreterPtr 结构体** (VmapInterpreter.h)：
- 包装器类，管理 Vmap 变换的解释器实例
- 存储指向基础 Interpreter 对象的指针
- 在构造时验证该解释器的类型确实是 Vmap

**核心方法**：

1. **processImpl** - 处理操作
   - 设置 TLS（线程本地存储）的分发键为 FuncTorchVmapMode
   - 调用操作的 boxed 版本（接受栈参数的通用形式）

2. **sendToNextInterpreterImpl** - 转发到下一个解释器
   - 如果动态层栈为空，进行栈的合理性检查
   - 再次通过 boxed 调用将操作派遣给下一个处理层

**元数据访问**：
- batchSize() - 获取批处理大小
- randomness() - 获取随机性配置

---

**关键作用**：
- Vmap 变换的运行时执行层
- 在操作分发链中充当中间层
- 通过 TLS 设置告诉系统当前在 Vmap 变换上下文中
- 支持分层解释器堆栈架构

---

**总结**：

• Vmap 变换的解释器封装，管理批向量化操作
• 两个主要方法处理操作：设置 TLS 后执行，或转发到下一个解释器
• 支持批大小和随机性配置的查询
• 遵循 functorch 的分层解释器堆栈设计模式
