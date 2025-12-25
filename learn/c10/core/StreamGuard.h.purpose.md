StreamGuard.h 定义了三个 RAII 类，用于管理 CUDA/设备流的生命周期：

**StreamGuard**
- 强制指定设备流，在作用域结束时恢复原始流和设备
- 禁止默认构造、拷贝和移动，确保流程可控
- 提供 `reset_stream()` 切换到新流，`original_stream()/current_stream()` 查询流状态
- 不推荐在算子定义中使用（应由调用者处理流管理）

**OptionalStreamGuard**
- StreamGuard 的可选版本，支持未初始化状态
- 可以用 `std::optional<Stream>` 构造，如果为 nullopt 则不设置流
- 提供 `reset()` 方法恢复到未初始化状态
- 适合不确定是否需要流管理的场景

**MultiStreamGuard**
- 一次性设置多个设备上的流
- 接受 `ArrayRef<Stream>` 参数
- 销毁时恢复所有设备的原始流

**核心特点：**
- 都基于 `InlineStreamGuard` 和 `VirtualGuardImpl` 实现
- 都禁止拷贝和移动操作（RAII 原则）
- 自动管理流/设备状态的保存和恢复
