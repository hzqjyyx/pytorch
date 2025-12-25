这个文件定义了一个后端选择的回退机制：

- **TORCH_LIBRARY_IMPL** 宏注册了一个库实现，用于处理所有操作符（`_`）和 `BackendSelect` 调度键
- **m.fallback()** 设置了一个回退函数，使用 `torch::CppFunction::makeFallthrough()` 创建
- **makeFallthrough()** 生成一个直通函数，让调度流程继续传递到下一个优先级的后端实现
- 作用是确保当特定后端没有实现某个操作时，能够优雅地降级到其他可用的后端实现
- 这是 PyTorch 多后端支持的核心机制之一，用于实现后端的灵活选择和自动回退
