## PythonFallbackKernel 文件功能分析

这两个文件实现了 PyTorch 的 Python 调度系统，用于处理在 Python 层面拦截和分发操作。

### 核心组件

**TLS 状态管理 (Thread Local Storage)**
- `tls_on_entry`: 跟踪进入 `__torch_dispatch__` 时的调度器状态
- 保证在执行 Python 代码时 TLS 为空，在返回时恢复状态
- `StashTLSOnEntryGuard`: RAII 守卫，安全地临时清空和恢复 TLS

**调度键集合**
- `after_Python_keyset`: 包含 Python 键之下的所有调度键，用于重新分发时排除 Python 键

**四个主要 Fallback 函数**

1. **`pythonFallback()`** (最核心)
   - 处理带 Python 调度键的操作
   - 优先使用 TorchDispatchMode 的 PyInterpreter
   - 否则从参数张量中查找 PyInterpreter
   - 若找不到解释器则断言失败

2. **`pythonDispatcherFallback()`**
   - 处理 PythonDispatcher 键
   - 从 TLS 获取状态，移除 PythonDispatcher 键后重新分发

3. **`pythonTLSSnapshotFallback()`**
   - 捕获 TLS 快照
   - 使用 `MaybeSetTLSOnEntryGuard` 安全地设置 TLS
   - 重新分发到 PythonTLSSnapshot 之下的键

4. **`preDispatchFallback()`**
   - 预调度钩子的无操作 Fallback
   - 直接重新分发到 PreDispatch 之下的键
   - 存在原因：PythonDispatcher 需要拦截点

**RAII 守卫类** (`at::impl`)
- `RestorePythonTLSSnapshot`: 保存并恢复 TLS 快照
- `MaybeSetTLSOnEntryGuard`: 条件性地设置 TLS（已设置则跳过）

---

### 功能总结

- **Python 调度拦截**: 通过 `TORCH_LIBRARY_IMPL` 注册四个调度键的 Fallback 处理器
- **解释器查找**: 从 TorchDispatchMode 或张量参数中定位 Python 解释器
- **TLS 管理**: 维护调度状态快照，确保 Python 代码执行时状态清洁
- **键重新分发**: 移除当前调度键，转发到后续处理链
- **模式支持**: 支持 TorchDispatchMode、PythonDispatcher、PreDispatch 等扩展机制
