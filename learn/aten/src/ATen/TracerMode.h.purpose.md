这个文件定义了 PyTorch 中 **tracing 模式的开关控制机制**。

## 核心功能

提供了三个关键 API 来控制 tracing 功能的启用/禁用：

1. **`is_dispatch_enabled()`** - 检查 `Tracer` dispatch key 是否启用
2. **`set_dispatch_enabled(bool)`** - 设置 `Tracer` dispatch key 的启用状态
3. **`NoTracerDispatchMode`** - RAII guard，在作用域内禁用 tracer dispatching

## 历史演变

**旧机制**（tracing 在 VariableType 中）：
- 通过 `AutoDispatchBelowADInplaceOrView` guard 控制
- 通过 `setTracingState()` API 控制 TLS 中的 `TracingState` 对象
- tracing 和 autograd 共享 `Autograd` dispatch key

**新机制**（tracing 独立出来后）：
- 引入独立的 `Tracer` dispatch key（默认关闭）
- 通过 `set_dispatch_enabled()` 控制 dispatch key
- 通过 `NoTracerDispatchMode` guard 替代旧的 `AutoDispatchBelowADInplaceOrView` 语义

## 启用条件

Tracing 功能启用需要同时满足：
1. TLS 中的 `TracingState` 对象非空
2. 已调用 `set_dispatch_enabled(true)`
3. 不在 `NoTracerDispatchMode` 作用域内

## 实现细节

- `is_dispatch_enabled()`: 检查 `Tracer` key 是否被 included 且未被 excluded
- `set_dispatch_enabled()`: 调用 `tls_set_dispatch_key_included()` 设置 TLS 状态，会断言当前不在 `NoTracerDispatchMode` 内
- `NoTracerDispatchMode`: 内部使用 `ExcludeDispatchKeyGuard` 排除 `Tracer` key

---

**相关但未详述的内容：**
- ROCm 相关：无
- Backward 相关：文件讨论了与 autograd 的历史关联，但主要是架构演变说明，不涉及具体 backward 实现
