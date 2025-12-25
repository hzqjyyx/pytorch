## ThreadLocalDebugInfo 功能分析

这是 PyTorch 的线程本地调试信息管理系统，允许在前向/后向传播过程中在不同层级间传递上下文信息。

### 核心设计

**基础类型**
- `DebugInfoKind` 枚举：定义调试信息类型（PRODUCER_INFO、MOBILE_RUNTIME_INFO、PROFILER_STATE、INFERENCE_CONTEXT、PARAM_COMMS_INFO 等）
- `DebugInfoBase`：基类，用于派生具体的调试信息对象
- `ThreadLocalDebugInfo`：线程本地存储的调试信息链表节点

**数据结构**
```
ThreadLocalDebugInfo (当前)
  ├─ info_: DebugInfoBase (具体信息)
  ├─ kind_: DebugInfoKind (信息类型)
  └─ parent_info_: ThreadLocalDebugInfo (上一级)
```
形成栈式结构，支持嵌套的调试上下文。

### 关键操作

| 函数 | 功能 |
|------|------|
| `get(kind)` | 从栈中查找指定类型的调试信息，向上遍历直到找到或返回 nullptr |
| `current()` | 获取当前栈顶的 ThreadLocalDebugInfo |
| `_push(kind, info)` | 压入新的调试信息，保存前一个状态为 parent_info_ |
| `_pop(kind)` | 弹出栈顶并验证类型，恢复 parent_info_ |
| `_peek(kind)` | 查看栈顶信息但不弹出 |
| `_forceCurrentDebugInfo(info)` | 强制替换整个栈（跨线程传播用）|

### DebugInfoGuard（RAII 模式）

- **作用**：作用域自动管理调试信息的压入/弹出
- **构造**：保存当前状态，压入新信息
- **析构**：恢复之前的状态
- **限制**：禁用拷贝和移动语义，确保正确的栈管理

### 使用场景

跨线程边界时，将主线程的调试信息传递给线程池线程（通过 `_forceCurrentDebugInfo`），使得观察者、日志、性能分析等工具能获取上层的模型 ID、上下文等信息。

---

**核心功能列表：**

- Thread-local 栈式调试信息管理
- 支持嵌套的作用域级调试上下文
- 线程安全的调试信息查询（`get`）
- RAII 风格的自动压栈/弹栈（`DebugInfoGuard`）
- 跨线程边界的调试信息传播（`_forceCurrentDebugInfo`）
- 类型检查的栈操作（`_pop`/`_peek` 验证类型匹配）
