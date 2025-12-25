## PyHandleCache 核心功能分析

**主要用途**：C++ 对象到 Python 对象的指针缓存机制，通过快速路径（cache hit）避免重复创建 Python 对象。

**使用条件**：
1. 必须是真正的缓存 - 缓存未命中时能通过其他方式重新获取对象
2. 必须是句柄 - Python 对象生命周期为静态，C++ 对象销毁时无需回收 Python 对象

**核心实现**：

`ptr_or()` 模板方法的三层逻辑：
1. **快速路径**（`C10_LIKELY`）：检查 PyInterpreter 匹配，直接返回缓存数据
2. **缓存初始化**：当缓存为空时，执行 `slow_accessor()` 获取真实对象，并尝试原子写入缓存
3. **缓存失效**：不同 interpreter 或竞态条件下，绕过缓存直接返回新对象

**内存安全机制**：
- 使用 `std::atomic<PyInterpreter*>` 保证多线程安全
- `memory_order_acquire/acq_rel` 确保内存可见性
- `compare_exchange_strong()` 实现无锁竞争避免

**性能考虑**：
- torchdeploy 下效能有限（多 interpreter 场景缓存利用率低）
- 潜在改进方案：每个 interpreter 一个缓存槽位

**关键特性**：
- 不维持强引用所有权
- 仅在单个解释器场景下高效
- GIL 保护假设

---

- **缓存机制**：快速路径避免频繁创建 Python 对象
- **线程安全**：原子操作 + 内存序保证
- **轻量级**：句柄语义，无所有权管理负担
- **多解释器支持**：带 interpreter 标签的缓存槽位机制
- **失效策略**：竞态时返回新对象而非等待
